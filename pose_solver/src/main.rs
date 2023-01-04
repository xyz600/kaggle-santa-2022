use lib::{array_solution::ArraySolution, solution::Solution};
use proconio::input;
use proconio::source::auto::AutoSource;
use std::collections::HashSet;
use std::io::Read;
use std::{
    collections::{BinaryHeap, HashMap},
    fs::File,
    path::PathBuf,
    str::FromStr,
};

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
        let x = (value % offset as u32) as i16 - (offset / 2 - 1);
        let y = (value / offset as u32) as i16 - (offset / 2 - 1);
        Coord { x, y }
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
                Coord::new(0, -64),
                Coord::new(0, 32),
                Coord::new(0, 16),
                Coord::new(0, 8),
                Coord::new(0, 4),
                Coord::new(0, 2),
                Coord::new(0, 1),
                Coord::new(0, 1),
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

    fn diff(&self, to: &Pose) -> DiffPose {
        let mut diff_list = [Direction::None; 8];
        for i in 0..8 {
            assert!(self.arm_list[i].x.abs_diff(to.arm_list[i].x) <= 1);
            assert!(self.arm_list[i].y.abs_diff(to.arm_list[i].y) <= 1);

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
                unreachable!()
            };
        }
        DiffPose { diff_list }
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
                ret.push(DiffPose {
                    diff_list: stack.clone(),
                })
            } else {
                let max_value = get_max_value(depth);
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
                } else {
                    [Direction::Left, Direction::Right, Direction::None]
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
}

#[cfg(test)]
mod tests {
    use crate::{Coord, Pose};

    #[test]
    fn test_encdec() {
        let pose = Pose {
            arm_list: [
                Coord::new(-128, 128),
                Coord::new(-64, 64),
                Coord::new(32, -32),
                Coord::new(16, 16),
                Coord::new(0, 0),
                Coord::new(4, 2),
                Coord::new(2, 1),
                Coord::new(1, 2),
            ],
        };
        let reconstructed = Pose::decode(pose.encode());
        for i in 0..8 {
            assert_eq!(reconstructed.arm_list[i].y, pose.arm_list[i].y);
        }
    }
}

fn calculate_cost(diff: &DiffPose, from: &Pose, image: &HashMap<(i16, i16), Cell>) -> i64 {
    let mut to = from.clone();
    to.apply(diff);

    let from_coord = from.coord();
    let from_cell = image[&from_coord];
    let to_coord = to.coord();
    let to_cell = image[&to_coord];

    let dr = (from_cell.r - to_cell.r).abs();
    let dg = (from_cell.g - to_cell.g).abs();
    let db = (from_cell.b - to_cell.b).abs();

    let color_cost = (dr + dg + db) * 3 * 10000;

    let no_move = diff
        .diff_list
        .iter()
        .filter(|&v| *v != Direction::None)
        .count();
    let pos_cost = ((no_move as f64).sqrt() * 10000.0 * 255.0).round() as i64;

    color_cost + pos_cost
}

fn calculate_lb_cost(from: &Pose, to_coord: &(i16, i16), image: &HashMap<(i16, i16), Cell>) -> i64 {
    let from_coord = from.coord();
    let from_cell = image[&from_coord];
    let to_cell = image[to_coord];

    let dr = (from_cell.r - to_cell.r).abs();
    let dg = (from_cell.g - to_cell.g).abs();
    let db = (from_cell.b - to_cell.b).abs();

    let color_cost = (dr + dg + db) * 3 * 10000;

    let dy = from_coord.0.abs_diff(to_coord.0) as i64;
    let dx = from_coord.1.abs_diff(to_coord.1) as i64;

    let pos_cost = (((dx + dy) as f64).sqrt() * 10000.0 * 255.0).round() as i64;

    color_cost + pos_cost
}

// dynamic dijkstra で最小コストルートを求める
fn next_move(
    init_pose: Pose,
    to: (i16, i16),
    image: &HashMap<(i16, i16), Cell>,
) -> (f64, Vec<DiffPose>) {
    let mut visited = HashMap::<u128, (i64, u128)>::new();

    let mut que = BinaryHeap::<(i64, i64, u128)>::new();
    que.push((0, 0, init_pose.encode()));

    eprintln!("from = {:?}, to = {:?}", init_pose.coord(), to);

    while let Some((_lb_cost_neg, cost, encoded)) = que.pop() {
        // to min-heap
        let cost = -cost;

        let pose = Pose::decode(encoded);
        let cur_enc = pose.encode();

        if pose.coord() == to {
            // 復元
            let from = init_pose.coord();
            let mut coord = to;
            let mut diff_list = vec![];
            let mut cur_pose = pose;

            while coord != from {
                let prev_pose_encoded = visited[&cur_pose.encode()].1;
                let prev_pose = Pose::decode(prev_pose_encoded);
                let diff = prev_pose.diff(&cur_pose);

                diff_list.push(diff);
                cur_pose = prev_pose;
                coord = cur_pose.coord();
            }

            return (cost as f64 / (255.0 * 10000.0), diff_list);
        }

        for diff in pose.next_diff_list() {
            let next_cost = cost + calculate_cost(&diff, &pose, image);

            let mut next_pose = pose.clone();
            next_pose.apply(&diff);
            let next_pose_encoded = next_pose.encode();

            if !visited.contains_key(&next_pose_encoded)
                || visited[&next_pose_encoded].0 > next_cost
            {
                visited.insert(next_pose_encoded, (next_cost, cur_enc));
                // コストの下限値を求めて A*
                let lb_cost = next_cost + calculate_lb_cost(&next_pose, &to, image);

                que.push((-lb_cost, -next_cost, next_pose_encoded));
            }
        }
    }

    unreachable!();
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

fn load_image(filepath: &PathBuf) -> HashMap<(i16, i16), Cell> {
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
    let cell_list = data
        .iter()
        .map(|(x, y, r, g, b)| Cell {
            x: *x,
            y: *y,
            r: (*r * 255.0).round() as i64,
            g: (*g * 255.0).round() as i64,
            b: (*b * 255.0).round() as i64,
        })
        .collect::<Vec<_>>();

    let mut ret = HashMap::new();
    for cell in cell_list.into_iter() {
        ret.insert((cell.y as i16, cell.x as i16), cell);
    }
    ret
}

fn main() {
    // solution を読み込む
    let solution = ArraySolution::load(&PathBuf::from_str("solution_snake.tsp").unwrap());
    // let solution = ArraySolution::load(&PathBuf::from_str("solution_split_lkh.tsp").unwrap());

    let mut index_table = HashMap::<(i16, i16), u32>::new();
    let mut rev_index_table = vec![(0, 0); solution.len()];
    {
        let mut index = 0u32;
        for y in (-128..=128).rev() {
            for x in -128..=128 {
                index_table.insert((y, x), index);
                rev_index_table[index as usize] = (y, x);
                index += 1;
            }
        }
    }
    let index_table = index_table;
    let rev_index_table = rev_index_table;

    let image = load_image(&PathBuf::from_str("data/image.csv").unwrap());

    // 初手から次の遷移までに要する最少コストを BFS で sequential に計算
    // 原点が最初
    let mut id = index_table[&(0, 0)];
    // 初期は [(0, -64), (0, 32), (0, 16), (0, 8), (0, 4), (0, 2), (0, 1), (0, 1)]
    let mut pose = Pose::new();

    let mut all_pose_list = vec![pose.clone()];

    let mut cost_all = 0.0;

    for iter in 0..solution.len() {
        let next_id = solution.next(id);
        let next_pos = rev_index_table[next_id as usize];
        let (cost, diff_list) = next_move(pose, next_pos, &image);
        for diff in diff_list {
            pose.apply(&diff);
            all_pose_list.push(pose.clone());
        }
        id = next_id;
        cost_all += cost;

        eprintln!(
            "iter = {} / {}, cost = {} ({})",
            iter,
            257 * 257,
            cost_all,
            cost
        );
    }

    // print final pose
}
