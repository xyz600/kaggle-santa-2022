use lib::evaluate::evaluate;
use lib::lkh::LKHConfig;
use lib::solution::{self, Solution};
use lib::{array_solution::ArraySolution, distance::DistanceFunction};
use lib::{distance, lkh};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::collections::{BinaryHeap, HashSet};
use std::thread;
use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    str::FromStr,
};

use crate::mst_solution::create_mst_solution;
use crate::tsp;
use crate::util::{load_image, Cell};

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
            (257 * 2 + 3 * 3) * 255 * 10000
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
    distance_table: Vec<Vec<i64>>,
}

impl StepDistanceFunction {
    fn new(
        pose_list: &Vec<Pose>,
        color_list: &Vec<Cell>,
    ) -> (StepDistanceFunction, Vec<HashSet<u32>>) {
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

        fn calculate_cost(from: u32, to: u32, diff: &DiffPose, color_list: &Vec<Cell>) -> i64 {
            // color cost
            let dr = color_list[from as usize].r - color_list[to as usize].r;
            let dg = color_list[from as usize].g - color_list[to as usize].g;
            let db = color_list[from as usize].b - color_list[to as usize].b;
            let color_cost = (dr.abs() + dg.abs() + db.abs()) * 10000;

            // position cost
            let pos_count = diff
                .diff_list
                .iter()
                .filter(|&v| *v != Direction::None)
                .count();
            let pos_cost = ((pos_count as f64).sqrt() * 255.0 * 10000.0) as i64;

            color_cost + pos_cost
        }

        // 全点間の最短手数を求める
        // 厳密コストを計算するので、dijkstra で計算
        let distance_table = (0..dim)
            .into_par_iter()
            .map(|from| {
                let mut que = BinaryHeap::new();
                let mut distance_row = vec![std::i64::MAX; dim];
                que.push((0i64, from as u32));
                distance_row[from] = 0;

                let mut counter = 0;

                while let Some((cost, id)) = que.pop() {
                    let cost = -cost;

                    // skip
                    if distance_row[id as usize] < cost {
                        continue;
                    }
                    counter += 1;

                    for n_id in neighbor_list[id as usize].iter() {
                        let pose1 = &pose_list[id as usize];
                        let pose2 = &pose_list[*n_id as usize];

                        let diff = pose1.diff(&pose2).unwrap();
                        let next_cost = cost + calculate_cost(id, *n_id, &diff, &color_list);

                        if distance_row[*n_id as usize] > next_cost {
                            distance_row[*n_id as usize] = next_cost;
                            que.push((-next_cost, *n_id));
                        }
                    }
                }
                assert_eq!(counter, dim);
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

        (StepDistanceFunction { distance_table }, neighbor_list)
    }
}

impl DistanceFunction for StepDistanceFunction {
    fn distance(&self, id1: u32, id2: u32) -> i64 {
        self.distance_table[id1 as usize][id2 as usize]
    }

    fn dimension(&self) -> u32 {
        self.distance_table.len() as u32
    }

    fn name(&self) -> String {
        "pose_nostep_function".to_string()
    }
}

pub fn solve() {
    const SIZE: usize = 257 * 257;
    let pose_list = calculate_pose_map(SIZE);
    let color_list = load_image(&PathBuf::from_str("./data/image.csv").unwrap());
    eprintln!("finish pose assignment");

    // 初期解を作るために、遷移手数を最適化
    let (distance, neighbor_list) = StepDistanceFunction::new(&pose_list, &color_list);
    eprintln!("finish calculate distance function");

    let init_solution = create_mst_solution(&distance, &neighbor_list);

    let solution = tsp::solve_tsp(&distance, init_solution, 1.0 / (255.0 * 10000.0));

    solution.save(&PathBuf::from_str("final_result_bfs.tsp").unwrap())
}
