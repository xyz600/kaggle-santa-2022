use lib::evaluate::evaluate;
use lib::lkh::LKHConfig;
use lib::solution::{self, Solution};
use lib::{array_solution::ArraySolution, distance::DistanceFunction};
use lib::{distance, lkh};
use proconio::input;
use proconio::source::auto::AutoSource;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    str::FromStr,
};

use crate::tsp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Coord {
    y: i16,
    x: i16,
}

impl Coord {
    fn new(y: i16, x: i16) -> Coord {
        Coord { y, x }
    }
    fn encode(&self, bit_width: i16) -> u32 {
        let offset = 1 << bit_width;
        let ey = (self.y + offset / 2) as u32 - 1;
        let ex = (self.x + offset / 2) as u32 - 1;
        ey * offset as u32 + ex
    }

    fn decode(value: u32, bit_width: i16) -> Coord {
        let offset = (1 << bit_width) as i16;
        let y = (value / offset as u32) as i16 - (offset / 2 - 1);
        let x = (value % offset as u32) as i16 - (offset / 2 - 1);
        Coord { y, x }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    Up,
    Left,
    Right,
    Down,
    None,
}

#[derive(Clone, Copy, Debug)]
struct DiffPose {
    diff_list: [Direction; 8],
}

#[derive(Clone, Copy, Debug)]
struct Pose {
    arm_list: [Coord; 8],
}

impl Pose {
    fn new() -> Pose {
        Pose {
            arm_list: [
                Coord::new(0, 64),
                Coord::new(0, -32),
                Coord::new(0, -16),
                Coord::new(0, -8),
                Coord::new(0, -4),
                Coord::new(0, -2),
                Coord::new(0, -1),
                Coord::new(0, -1),
            ],
        }
    }

    fn apply(&mut self, diff: &DiffPose) {
        for i in 0..8 {
            match diff.diff_list[i] {
                Direction::Up => self.arm_list[i].y += 1,
                Direction::Left => self.arm_list[i].x -= 1,
                Direction::Right => self.arm_list[i].x += 1,
                Direction::Down => self.arm_list[i].y -= 1,
                Direction::None => {}
            }
        }
    }

    fn encode(&self) -> u128 {
        let mut ret = 0u128;
        let bit_width_list = [9, 8, 7, 6, 5, 4, 3, 3];
        for i in 0..8 {
            ret <<= 2 * bit_width_list[i];
            ret += self.arm_list[i].encode(bit_width_list[i]) as u128;
        }
        ret
    }

    fn decode(mut value: u128) -> Pose {
        let mut arm_list = [Coord::new(0, 0); 8];
        let bit_width_list = [9, 8, 7, 6, 5, 4, 3, 3];

        for i in (0..8).rev() {
            let mask = (1 << (bit_width_list[i] * 2)) - 1;
            let coord_value = (value & mask) as u32;
            arm_list[i] = Coord::decode(coord_value, bit_width_list[i]);
            value >>= bit_width_list[i] * 2;
        }
        Pose { arm_list }
    }

    fn coord(&self) -> (i16, i16) {
        let mut y = 0;
        let mut x = 0;
        for i in 0..8 {
            y += self.arm_list[i].y;
            x += self.arm_list[i].x;
        }
        (y, x)
    }

    fn diff(&self, to: &Pose) -> Option<DiffPose> {
        let mut diff_list = [Direction::None; 8];
        for i in 0..8 {
            let cy = self.arm_list[i].y;
            let cx = self.arm_list[i].x;
            let ny = to.arm_list[i].y;
            let nx = to.arm_list[i].x;

            diff_list[i] = if cy + 1 == ny && cx == nx {
                Direction::Up
            } else if cy - 1 == ny && cx == nx {
                Direction::Down
            } else if cx + 1 == nx && cy == ny {
                Direction::Right
            } else if cx - 1 == nx && cy == ny {
                Direction::Left
            } else if cx == nx && cy == ny {
                Direction::None
            } else {
                return None;
            };
        }
        Some(DiffPose { diff_list })
    }

    fn can_reach_coord(&self, target_coord: (i16, i16)) -> bool {
        let ((min_y, min_x), (max_y, max_x)) = self.one_step_range();

        let (y, x) = target_coord;
        min_y <= y && y <= max_y && min_x <= x && x <= max_x
    }

    fn direction_candidate(&self, i: usize) -> [Direction; 3] {
        let max_value_list = [64, 32, 16, 8, 4, 2, 1, 1];
        let max_value = max_value_list[i];
        // stack に積まれた方向の分も加味
        let y = self.arm_list[i].y;
        let x = self.arm_list[i].x;

        if x == -max_value && y == -max_value {
            [Direction::Up, Direction::None, Direction::Right]
        } else if x == max_value && y == -max_value {
            [Direction::Up, Direction::None, Direction::Left]
        } else if x == -max_value && y == max_value {
            [Direction::Down, Direction::None, Direction::Right]
        } else if x == max_value && y == max_value {
            [Direction::Down, Direction::None, Direction::Left]
        } else if x.abs() == max_value {
            [Direction::Up, Direction::Down, Direction::None]
        } else if y.abs() == max_value {
            [Direction::Left, Direction::Right, Direction::None]
        } else {
            unreachable!()
        }
    }

    fn one_step_range(&self) -> ((i16, i16), (i16, i16)) {
        let coord = self.coord();
        let mut min_x = coord.1;
        let mut max_x = coord.1;
        let mut min_y = coord.0;
        let mut max_y = coord.0;

        for i in 0..8 {
            for dir in self.direction_candidate(i) {
                match dir {
                    Direction::Up => max_y += 1,
                    Direction::Left => min_x -= 1,
                    Direction::Right => max_x += 1,
                    Direction::Down => min_y -= 1,
                    Direction::None => {}
                }
            }
        }

        ((min_y, min_x), (max_y, max_x))
    }

    fn next_diff_list(&self) -> Vec<DiffPose> {
        fn get_max_value(depth: usize) -> i16 {
            let ret = [64, 32, 16, 8, 4, 2, 1, 1];
            ret[depth]
        }

        fn inner(
            pos_list: &[Coord; 8],
            depth: usize,
            ret: &mut Vec<DiffPose>,
            stack: &mut [Direction; 8],
        ) {
            if depth == 8 {
                // 全ての diff を積み終わった

                let mut changed = false;
                for i in 0..8 {
                    if stack[i] != Direction::None {
                        changed = true;
                        break;
                    }
                }
                // 全て None を避ける
                if changed {
                    ret.push(DiffPose {
                        diff_list: stack.clone(),
                    })
                }
            } else {
                let max_value = get_max_value(depth);
                // stack に積まれた方向の分も加味
                let y = pos_list[depth].y;
                let x = pos_list[depth].x;

                let candidate_dirs = if x == -max_value && y == -max_value {
                    [Direction::Up, Direction::None, Direction::Right]
                } else if x == max_value && y == -max_value {
                    [Direction::Up, Direction::None, Direction::Left]
                } else if x == -max_value && y == max_value {
                    [Direction::Down, Direction::None, Direction::Right]
                } else if x == max_value && y == max_value {
                    [Direction::Down, Direction::None, Direction::Left]
                } else if x.abs() == max_value {
                    [Direction::Up, Direction::Down, Direction::None]
                } else if y.abs() == max_value {
                    [Direction::Left, Direction::Right, Direction::None]
                } else {
                    unreachable!()
                };

                for dir in candidate_dirs {
                    stack[depth] = dir;
                    inner(pos_list, depth + 1, ret, stack);
                }
            }
        }

        let mut ret = vec![];
        let mut stack = [Direction::None; 8];
        inner(&self.arm_list, 0, &mut ret, &mut stack);
        ret
    }

    fn to_string(&self) -> String {
        let mut vs = vec![];
        for i in 0..8 {
            vs.push(format!("{} {}", self.arm_list[i].x, self.arm_list[i].y));
        }
        vs.join(";") + "\n"
    }
}

fn calculate_pose_map(dimension: usize) -> Vec<Pose> {
    let init_pose = Pose::new();

    let mut index_table = HashMap::<(i16, i16), u32>::new();
    {
        let mut index = 0u32;
        for y in (-128..=128).rev() {
            for x in -128..=128 {
                index_table.insert((y, x), index);
                index += 1;
            }
        }
    }
    let index_table = index_table;

    let mut ret = vec![Pose::new(); dimension];
    let mut visited = vec![false; dimension];
    visited[index_table[&(0, 0)] as usize] = true;

    // bfs で姿勢付与
    let mut que = VecDeque::new();
    que.push_back(init_pose.encode());

    while let Some(pose_encoded) = que.pop_front() {
        let pose = Pose::decode(pose_encoded);
        let pose_coord = pose.coord();
        let coord_index = index_table[&pose_coord] as usize;
        visited[coord_index] = true;

        for diff in pose.next_diff_list() {
            let mut new_pose = pose.clone();
            new_pose.apply(&diff);
            let new_coord = new_pose.coord();
            let new_coord_index = index_table[&new_coord] as usize;
            if !visited[new_coord_index] {
                visited[new_coord_index] = true;
                ret[new_coord_index] = new_pose;
                que.push_back(new_pose.encode());
            }
        }
    }
    ret
}

macro_rules! input_fromfile {
    (path: $path:expr, $($t:tt)+) => {
        fn read_all(filepath: &PathBuf) -> String {
            let mut f = File::open(filepath).expect("file not found");
            let mut contents = String::new();

            f.read_to_string(&mut contents)
                .expect("something went wrong reading the file");

            contents
        }
        let contents = read_all($path);
        let source = AutoSource::from(contents.as_str());

        input! {
            from source,
            $($t)*
        }
    };
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    x: i64,
    y: i64,
    r: i64,
    g: i64,
    b: i64,
}

fn load_image(filepath: &PathBuf) -> Vec<Cell> {
    const SIZE: usize = 257;
    const LEN: usize = SIZE * SIZE;

    input_fromfile! {
        path: filepath,
        _1: String,
        _2: String,
        _3: String,
        _4: String,
        _5: String,
        data: [(i64, i64, f64, f64, f64); LEN],
    };
    data.iter()
        .map(|(x, y, r, g, b)| Cell {
            x: *x,
            y: *y,
            r: (*r * 255.0).round() as i64,
            g: (*g * 255.0).round() as i64,
            b: (*b * 255.0).round() as i64,
        })
        .collect::<Vec<_>>()
}

struct PoseDistanceFunction {
    pose_list: Vec<Pose>,
    color_table: Vec<Cell>,
    name: String,
}

impl PoseDistanceFunction {
    fn new(filepath: &PathBuf, pose_list: Vec<Pose>, name: String) -> PoseDistanceFunction {
        let color_table = load_image(filepath);
        PoseDistanceFunction {
            pose_list,
            name,
            color_table,
        }
    }
}

impl DistanceFunction for PoseDistanceFunction {
    // 1手で到達できる場合のみまともな姿勢を付与する
    fn distance(&self, id1: u32, id2: u32) -> i64 {
        let pose1 = &self.pose_list[id1 as usize];
        let pose2 = &self.pose_list[id2 as usize];

        if let Some(diff) = pose1.diff(&pose2) {
            let c1 = &self.color_table[id1 as usize];
            let c2 = &self.color_table[id2 as usize];
            let dr = (c1.r - c2.r).abs();
            let dg = (c1.g - c2.g).abs();
            let db = (c1.b - c2.b).abs();

            let dist = (diff
                .diff_list
                .iter()
                .filter(|&v| *v != Direction::None)
                .count() as f64)
                .sqrt();
            (dist * 10000.0 * 255.0) as i64 + (dr + dg + db) * 10000 * 3
        } else {
            std::i64::MAX / (self.pose_list.len() as i64 * 4)
        }
    }

    fn dimension(&self) -> u32 {
        self.pose_list.len() as u32
    }

    fn name(&self) -> String {
        format!("pose_distance_function_{}", self.name)
    }
}

fn reconstruct_route(route_str: Vec<char>) -> Vec<u32> {
    let mut index_table = HashMap::<(i32, i32), u32>::new();
    {
        let mut index = 0u32;
        for y in (-128..=128).rev() {
            for x in -128..=128 {
                index_table.insert((y, x), index);
                index += 1;
            }
        }
    }

    let mut y = 0;
    let mut x = 0;
    let mut solution_array = vec![index_table[&(y, x)]];

    for ch in route_str {
        if ch == 'U' {
            y += 1;
        } else if ch == 'R' {
            x += 1;
        } else if ch == 'D' {
            y -= 1;
        } else if ch == 'L' {
            x -= 1;
        } else {
            unreachable!()
        }
        let index = index_table[&(y, x)];
        solution_array.push(index);
    }
    solution_array
}

struct StepDistanceFunction {
    distance_table: Vec<Vec<u32>>,
}

impl StepDistanceFunction {
    fn new(pose_list: &Vec<Pose>) -> StepDistanceFunction {
        // 8x8 の領域を見て、遷移できる場所を確認

        let dim = pose_list.len();

        let mut index_table = HashMap::<(i16, i16), u32>::new();
        {
            let mut index = 0u32;
            for y in (-128..=128).rev() {
                for x in -128..=128 {
                    index_table.insert((y, x), index);
                    index += 1;
                }
            }
        }
        let index_table = index_table;

        // 対称性確認
        (0..dim).into_par_iter().for_each(|from| {
            for to in from + 1..dim {
                let maybe_ft = pose_list[from].diff(&pose_list[to]);
                let maybe_tf = pose_list[to].diff(&pose_list[from]);
                assert_eq!(maybe_ft.is_some(), maybe_tf.is_some());

                // 逆の動きをしてたどり着くことができる
                if let Some(ft) = maybe_ft {
                    let tf = maybe_tf.unwrap();

                    let opposite = |dir1: Direction, dir2: Direction| match dir1 {
                        Direction::Up => dir2 == Direction::Down,
                        Direction::Left => dir2 == Direction::Right,
                        Direction::Right => dir2 == Direction::Left,
                        Direction::Down => dir2 == Direction::Up,
                        Direction::None => dir2 == Direction::None,
                    };

                    for i in 0..8 {
                        assert!(opposite(ft.diff_list[i], tf.diff_list[i]));
                    }
                }
            }
        });

        eprintln!("finish checking diff symmetric");

        // 1手で進める近傍リスト
        let neighbor_list = (0..pose_list.len())
            .into_par_iter()
            .map(|i| {
                let coord = pose_list[i].coord();
                let mut neighbor = HashSet::new();
                for dy in -8..=8 {
                    for dx in -8..=8 {
                        let target_coord = (coord.0 + dy, coord.1 + dx);
                        if coord != target_coord
                            && (-128 <= target_coord.0 && target_coord.0 <= 128)
                            && (-128 <= target_coord.1 && target_coord.1 <= 128)
                        {
                            let target_id = index_table[&target_coord];
                            if let Some(_diff) = pose_list[i].diff(&pose_list[target_id as usize]) {
                                neighbor.insert(target_id);
                            }
                        }
                    }
                }
                neighbor
            })
            .collect::<Vec<_>>();

        // 近傍の対称性確認
        (0..dim).into_par_iter().for_each(|from| {
            for to in 0..dim {
                if neighbor_list[from].contains(&(to as u32)) {
                    neighbor_list[to].contains(&(from as u32));
                }
            }
        });

        eprintln!("finish checking neighbor symmetric.");

        // 全点間の最短手数を求める
        // 距離1なので bfs で ok

        let distance_table = (0..dim)
            .into_par_iter()
            .map(|from| {
                let mut que = VecDeque::new();
                let mut distance_row = vec![std::u32::MAX; dim];
                que.push_back((0u32, from as u32));
                distance_row[from] = 0;

                while let Some((depth, id)) = que.pop_front() {
                    for n_id in neighbor_list[id as usize].iter() {
                        if distance_row[*n_id as usize] > depth + 1 {
                            distance_row[*n_id as usize] = depth + 1;
                            que.push_back((depth + 1, *n_id));
                        }
                    }
                }

                assert_eq!(distance_row[from], 0);
                for n_id in neighbor_list[from].iter() {
                    assert_eq!(distance_row[*n_id as usize], 1);
                }

                distance_row
            })
            .collect::<Vec<_>>();

        eprintln!("finish create distance function");

        (0..dim).into_par_iter().for_each(|from| {
            for to in from..dim {
                let ft = distance_table[from][to];
                let tf = distance_table[to][from];
                assert_eq!(tf, ft);
            }
        });
        eprintln!("check distance_table symmetry validation");

        StepDistanceFunction { distance_table }
    }
}

impl DistanceFunction for StepDistanceFunction {
    fn distance(&self, id1: u32, id2: u32) -> i64 {
        self.distance_table[id1 as usize][id2 as usize] as i64
    }

    fn dimension(&self) -> u32 {
        self.distance_table.len() as u32
    }

    fn name(&self) -> String {
        "pose_nostep_function".to_string()
    }
}

fn get_cache_filepath(distance: &impl DistanceFunction) -> PathBuf {
    PathBuf::from_str(format!("{}.cache", distance.name()).as_str()).unwrap()
}

struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> UnionFind {
        UnionFind {
            parent: (0..size).collect(),
            size: vec![1; size],
        }
    }

    fn find(&mut self, child: usize) -> usize {
        if self.parent[child] != child {
            let parent = self.find(self.parent[child]);
            self.parent[child] = parent;
        }
        self.parent[child]
    }

    fn unite(&mut self, v1: usize, v2: usize) {
        let p1 = self.find(v1);
        let p2 = self.find(v2);
        if self.size[p1] < self.size[p2] {
            self.parent[p1] = self.parent[p2];
            self.size[p2] += self.size[p1];
        } else {
            self.parent[p2] = self.parent[p1];
            self.size[p1] += self.size[p2];
        }
    }

    fn same(&mut self, v1: usize, v2: usize) -> bool {
        self.find(v1) == self.find(v2)
    }

    fn size(&mut self, v: usize) -> usize {
        let parent = self.find(v);
        self.size[parent]
    }
}

fn create_mst_solution(distance: &impl DistanceFunction) -> ArraySolution {
    let dim = distance.dimension() as usize;
    let mut distance_list = vec![vec![]; dim];
    let mut uf = UnionFind::new(dim);

    for from in 0..dim {
        for to in from + 1..dim {
            let dist = distance.distance(from as u32, to as u32);
            distance_list[dist as usize].push((from, to));
        }
    }

    eprintln!("finish edge list.");

    let mut counter = 0;
    let mut edges = vec![];
    'outer: while counter < dim {
        for dist in 0..dim {
            for (from, to) in distance_list[dist].iter() {
                if !uf.same(*from, *to) {
                    uf.unite(*from, *to);
                    edges.push((*from, *to));
                    counter += 1;
                    if counter == dim - 1 {
                        break 'outer;
                    }
                }
            }
        }
    }

    eprintln!("finish creating mst.");

    // mst から dfs で経路を復元
    // 適当に root を決めて、dfs で潜ってルートを決定
    let mut neighbor_table = vec![vec![]; dim];
    for (from, to) in edges.into_iter() {
        neighbor_table[from].push(to as u32);
        neighbor_table[to].push(from as u32);
    }

    fn inner(
        current: u32,
        visited: &mut Vec<bool>,
        seq: &mut Vec<u32>,
        neighbor_table: &Vec<Vec<u32>>,
    ) {
        seq.push(current);

        for next in neighbor_table[current as usize].iter() {
            if !visited[*next as usize] {
                visited[*next as usize] = true;
                inner(*next, visited, seq, neighbor_table);
            }
        }
    }

    let mut visited = vec![false; dim];
    visited[0] = true;
    let mut seq = vec![];
    inner(0, &mut visited, &mut seq, &neighbor_table);

    eprintln!("finish reconstruct solution");

    ArraySolution::from_array(seq)
}

pub fn solve() {
    const SIZE: usize = 257 * 257;
    let pose_list = calculate_pose_map(SIZE);
    eprintln!("finish pose assignment");

    // 初期解を作るために、遷移手数を最適化
    let distance = StepDistanceFunction::new(&pose_list);
    eprintln!("finish calculate distance function");

    let init_solution = create_mst_solution(&distance);
    // let init_solution = ArraySolution::load(&PathBuf::from_str("./solution_all_lkh.tsp").unwrap());

    let solution = tsp::solve_tsp(&distance, init_solution);

    solution.save(&PathBuf::from_str("final_result_bfs.tsp").unwrap())
}
