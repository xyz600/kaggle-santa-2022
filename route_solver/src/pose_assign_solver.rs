use lib::{array_solution::ArraySolution, solution::Solution};
use proconio::input;
use proconio::source::auto::AutoSource;
use std::io::{BufWriter, Read, Write};
use std::{
    collections::{BinaryHeap, HashMap},
    fs::File,
    path::PathBuf,
    str::FromStr,
};

use crate::util::{DiffPose, Direction, Pose};

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
    let mut visited = HashMap::<u128, (i64, i64, u128)>::new();

    let mut que = BinaryHeap::<(i64, i64, u128, u128)>::new();
    que.push((0, 0, init_pose.encode(), init_pose.encode()));

    while let Some((_lb_cost, cost, encoded, prev_encoded)) = que.pop() {
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
                let prev_pose_encoded = visited[&cur_pose.encode()].2;
                let prev_pose = Pose::decode(prev_pose_encoded);
                let diff = prev_pose.diff(&cur_pose);

                diff_list.push(diff);
                cur_pose = prev_pose;
                coord = cur_pose.coord();
            }
            diff_list.reverse();
            return (cost as f64 / (255.0 * 10000.0), diff_list);
        }

        // prev 登録が正しい物を信じる
        if visited.contains_key(&cur_enc) && visited[&cur_enc].2 != prev_encoded {
            continue;
        }

        for diff in pose.next_diff_list() {
            let next_cost = cost + calculate_cost(&diff, &pose, image);

            let mut next_pose = pose.clone();
            next_pose.apply(&diff);
            let next_pose_encoded = next_pose.encode();
            // コストの下限値を求めて A*
            let lb_cost = next_cost + calculate_lb_cost(&next_pose, &to, image);

            if !visited.contains_key(&next_pose_encoded)
                || (
                    -visited[&next_pose_encoded].0,
                    -visited[&next_pose_encoded].1,
                ) < (-lb_cost, -next_cost)
            {
                visited.insert(next_pose_encoded, (lb_cost, next_cost, cur_enc));

                // 正確なコスト計算のために next_cost もいれる
                que.push((-lb_cost, -next_cost, next_pose_encoded, cur_enc));
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

fn save_pos_list(pose_list: &Vec<Pose>, filepath: &PathBuf) {
    let f = File::create(filepath).unwrap();
    let mut writer = BufWriter::new(f);

    writer.write("configuration".as_bytes()).unwrap();
    for pose in pose_list.iter() {
        writer.write(pose.to_string().as_bytes()).unwrap();
    }
}

pub fn solve() {
    // solution を読み込む
    let solution = ArraySolution::load(&PathBuf::from_str("solution_split_lkh.tsp").unwrap());

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
    let init_pose = Pose::new();

    let target_id = index_table[&(128, 128)];
    let mut cost_all = 0.0;

    let mut all_pose_list = vec![init_pose.clone()];
    let mut front_count = 0;
    {
        // 初手から次の遷移までに要する最少コストを BFS で sequential に計算
        // 原点が最初
        let mut id = index_table[&(0, 0)];
        // 初期は [(0, -64), (0, 32), (0, 16), (0, 8), (0, 4), (0, 2), (0, 1), (0, 1)]
        let mut pose = init_pose;

        for iter in 0..solution.len() {
            let next_id = solution.next(id);
            let next_pos = rev_index_table[next_id as usize];

            let (cost, diff_list) = next_move(pose, next_pos, &image);
            for diff in diff_list {
                eprintln!("  {:?}", diff);
                pose.apply(&diff);
                all_pose_list.push(pose.clone());
            }
            id = next_id;
            cost_all += cost;
            front_count += 1;

            if id == target_id {
                break;
            }

            eprintln!(
                "iter = {} / {}, cost = {} ({})",
                iter,
                257 * 257,
                cost_all,
                cost
            );
            eprintln!("pose = {:?}", pose.arm_list);
        }
    }
    let front_count = front_count;

    {
        let mut rev_all_pose_list = vec![init_pose.clone()];
        let mut pose = init_pose;
        let mut id = index_table[&(0, 0)];
        for iter in 0..solution.len() {
            let prev_id = solution.prev(id);
            let prev_pos = rev_index_table[prev_id as usize];

            let (cost, diff_list) = next_move(pose, prev_pos, &image);
            for diff in diff_list {
                pose.apply(&diff);
                rev_all_pose_list.push(pose.clone());
            }
            id = prev_id;
            cost_all += cost;

            if id == target_id {
                break;
            }

            eprintln!(
                "iter = {} / {}, cost = {} ({})",
                iter + front_count,
                257 * 257,
                cost_all,
                cost
            );
            eprintln!("pose = {:?}", pose.arm_list);
        }

        rev_all_pose_list.pop();
        rev_all_pose_list.reverse();
        all_pose_list.append(&mut rev_all_pose_list);
    }
    save_pos_list(
        &all_pose_list,
        &PathBuf::from_str("submission.csv").unwrap(),
    );
}
