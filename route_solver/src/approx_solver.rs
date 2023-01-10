use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    str::FromStr,
};

use lib::{
    array_solution::ArraySolution,
    distance::DistanceFunction,
    divide_and_conqure_solver::{self, DivideAndConqureConfig},
    evaluate::evaluate,
    lkh::{self, LKHConfig},
    neighbor_table::NeighborTable,
    opt3::{self, Opt3Config},
    solution::Solution,
};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::{
    mst_solution::create_mst_solution,
    onestep_distance::OneStepDistanceFunction,
    util::{DiffPose, Direction, Pose},
};

// s-t を同一視するため、以下のルールを採用
// - s-t をまとめて 1頂点として採用(p とする)
// - p と p 以外(q とする)の距離を測る時、s-q, t-q 間の近い方を採用
// - p と q が離れている限りは
struct SubPathDistance {
    coord_set: Vec<(i16, i16)>,
    begin_id: u32,
    end_id: u32,
    orig_distance: OneStepDistanceFunction,
    orig_index_map: HashMap<(i16, i16), u32>,
    name: String,
}

impl SubPathDistance {}

impl DistanceFunction for SubPathDistance {
    fn distance(&self, id1: u32, id2: u32) -> i64 {
        if (self.begin_id == id1 && self.end_id == id2)
            || (self.begin_id == id2 && self.end_id == id1)
        {
            // s-t 間の距離は0に設定
            0
        } else {
            let orig_id1 = self.orig_index_map[&self.coord_set[id1 as usize]];
            let orig_id2 = self.orig_index_map[&self.coord_set[id2 as usize]];
            self.orig_distance.distance(orig_id1, orig_id2)
        }
    }

    fn dimension(&self) -> u32 {
        self.coord_set.len() as u32
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}

fn calculate_subpath(
    coord_list: Vec<(i16, i16)>,
    begin: (i16, i16),
    end: (i16, i16),
    orig_index_map: &HashMap<(i16, i16), u32>,
    name: String,
    is_64x64_vertical: bool,
) -> Vec<(i16, i16)> {
    let orig_distance = OneStepDistanceFunction::load(
        &PathBuf::from_str("data/image.csv").unwrap(),
        is_64x64_vertical,
    );

    let mut index_map = HashMap::new();
    for (i, coord) in coord_list.iter().enumerate() {
        index_map.insert(coord, i as u32);
    }
    let index_map = index_map;

    let mut begin_id = std::u32::MAX;
    let mut end_id = std::u32::MAX;
    for i in 0..coord_list.len() {
        if coord_list[i] == begin {
            begin_id = i as u32;
        }
        if coord_list[i] == end {
            end_id = i as u32;
        }
    }
    assert_ne!(begin_id, std::u32::MAX);
    assert_ne!(end_id, std::u32::MAX);
    let begin_id = begin_id;
    let end_id = end_id;

    let distance = SubPathDistance {
        coord_set: coord_list.clone(),
        begin_id,
        end_id,
        orig_distance,
        orig_index_map: orig_index_map.clone(),
        name,
    };

    // ±1のグリッド内部に含まれる近傍で mst
    let mut neighbor_list = vec![HashSet::new(); distance.dimension() as usize];
    for (i, &(y, x)) in coord_list.iter().enumerate() {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let ny = y + dy;
                let nx = x + dx;
                if index_map.contains_key(&(ny, nx)) {
                    let n_index = index_map[&(ny, nx)];
                    neighbor_list[i].insert(n_index);
                }
            }
        }
    }

    let init_solution = create_mst_solution(&distance, &neighbor_list);
    let subpath_solution = solve_tsp(
        &distance,
        init_solution,
        1.0 / (255.0 * 10000.0),
        begin_id,
        end_id,
    );

    let mut ret = vec![];
    let mut id = begin_id;
    if subpath_solution.prev(begin_id) == end_id {
        for _iter in 0..subpath_solution.len() {
            ret.push(coord_list[id as usize]);
            id = subpath_solution.next(id);
        }
    } else if subpath_solution.next(begin_id) == end_id {
        for _iter in 0..subpath_solution.len() {
            ret.push(coord_list[id as usize]);
            id = subpath_solution.prev(id);
        }
    } else {
        unreachable!();
    }
    ret
}

fn extract_upper_left() -> Vec<(i16, i16)> {
    let mut ret = vec![];
    for y in 64..=128 {
        for x in -128..=0 {
            ret.push((y, x));
        }
    }
    for y in 0..=63 {
        for x in -128..=-1 {
            ret.push((y, x));
        }
    }
    ret
}

fn extract_lower_left() -> Vec<(i16, i16)> {
    let mut ret = vec![];
    ret.push((0, -128));
    for y in -128..=-1 {
        for x in -128..=0 {
            ret.push((y, x));
        }
    }
    ret
}

fn extract_lower_right() -> Vec<(i16, i16)> {
    let mut ret = vec![];
    ret.push((-128, 0));
    for y in -128..=0 {
        for x in 1..=128 {
            ret.push((y, x));
        }
    }
    ret
}

fn extract_upper_right() -> Vec<(i16, i16)> {
    let mut ret = vec![];
    ret.push((0, 128));
    for y in 64..=128 {
        for x in 1..=128 {
            ret.push((y, x));
        }
    }
    for y in 1..=63 {
        for x in 2..=128 {
            ret.push((y, x));
        }
    }
    ret
}

fn get_cache_filepath(distance: &impl DistanceFunction) -> PathBuf {
    PathBuf::from_str(format!("{}.cache", distance.name()).as_str()).unwrap()
}

fn solve_tsp(
    distance: &(impl DistanceFunction + std::marker::Sync),
    mut init_solution: ArraySolution,
    scale: f64,
    begin: u32,
    end: u32,
) -> ArraySolution {
    // 最初に制約を満たすように変異を加える
    let neighbor_table = NeighborTable::load(&get_cache_filepath(distance));
    let init_solution = if init_solution.next(begin) != end && init_solution.prev(begin) != end {
        let success = lkh::connect(distance, &mut init_solution, &neighbor_table, begin, end, 8);
        if success {
            eprintln!("kick need and success.");
            init_solution.validate();
            assert!(init_solution.next(begin) == end || init_solution.prev(begin) == end);

            eprintln!(
                "kick eval = {}",
                evaluate(distance, &init_solution) as f64 * scale
            );
        }
        init_solution
    } else {
        init_solution
    };

    let mut best_solution = init_solution.clone();
    let mut best_eval = evaluate(distance, &best_solution);

    // cache 作成用
    opt3::solve(
        distance,
        init_solution.clone(),
        Opt3Config {
            use_neighbor_cache: true,
            cache_filepath: get_cache_filepath(distance),
            debug: false,
            neighbor_create_parallel: true,
            scale,
        },
    );

    let solutions = (0..100)
        .into_par_iter()
        .map(|_iter| {
            let local_solution = opt3::solve(
                distance,
                init_solution.clone(),
                Opt3Config {
                    use_neighbor_cache: true,
                    cache_filepath: get_cache_filepath(distance),
                    debug: false,
                    neighbor_create_parallel: false,
                    scale,
                },
            );

            let mut local_solution = lkh::solve(
                distance,
                local_solution,
                LKHConfig {
                    use_neighbor_cache: true,
                    cache_filepath: get_cache_filepath(distance),
                    debug: false,
                    time_ms: 60_000,
                    start_kick_step: 30,
                    kick_step_diff: 10,
                    end_kick_step: distance.dimension() as usize / 10,
                    fail_count_threashold: 50,
                    max_depth: 6,
                    neighbor_create_parallel: true,
                    scale,
                },
            );

            if local_solution.next(begin) != end && local_solution.prev(begin) != end {
                let success = lkh::connect(
                    distance,
                    &mut local_solution,
                    &neighbor_table,
                    begin,
                    end,
                    8,
                );
                if success {
                    local_solution.validate();
                    assert!(local_solution.next(begin) == end || local_solution.prev(begin) == end);
                }
            }

            for _iter in 1..3 {
                local_solution = divide_and_conqure_solver::solve(
                    distance,
                    &local_solution,
                    DivideAndConqureConfig {
                        no_split: 12,
                        debug: false,
                        time_ms: 30_000,
                        start_kick_step: 30,
                        kick_step_diff: 10,
                        end_kick_step: distance.dimension() as usize / 10,
                        fail_count_threashold: 50,
                        max_depth: 7,
                        scale,
                    },
                );
            }

            if local_solution.next(begin) != end && local_solution.prev(begin) != end {
                let success = lkh::connect(
                    distance,
                    &mut local_solution,
                    &neighbor_table,
                    begin,
                    end,
                    8,
                );
                if success {
                    local_solution.validate();
                    assert!(local_solution.next(begin) == end || local_solution.prev(begin) == end);
                }
            }

            let eval = evaluate(distance, &local_solution);
            eprintln!("eval = {}", eval as f64 / (255.0 * 10000.0));
            (eval, local_solution)
        })
        .collect::<Vec<_>>();

    for (eval, local_solution) in solutions.into_iter() {
        if best_eval > eval {
            best_solution = local_solution;
        }
    }

    eprintln!("finish opt3.");
    eprintln!(
        "eval = {}",
        evaluate(distance, &best_solution) as f64 * scale
    );
    best_solution.save(
        &PathBuf::from_str(format!("solution_opt3_{}.tsp", distance.name()).as_str()).unwrap(),
    );

    // 分割して並列化

    for iter in 1.. {
        best_solution = divide_and_conqure_solver::solve(
            distance,
            &best_solution,
            DivideAndConqureConfig {
                no_split: 12,
                debug: false,
                time_ms: 30_000,
                start_kick_step: 30,
                kick_step_diff: 10,
                end_kick_step: distance.dimension() as usize / 10,
                fail_count_threashold: 50,
                max_depth: 7,
                scale,
            },
        );
        let eval = evaluate(distance, &best_solution);
        eprintln!("finish splited lkh {} times.", iter);
        eprintln!("eval = {}", eval as f64 * scale);
        if best_eval == eval {
            break;
        } else {
            best_solution.save(
                &PathBuf::from_str(format!("solution_split_lkh_{}.tsp", distance.name()).as_str())
                    .unwrap(),
            );
        }
        best_eval = eval;
    }

    if best_solution.next(begin) != end && best_solution.prev(begin) != end {
        let success = lkh::connect(distance, &mut best_solution, &neighbor_table, begin, end, 8);
        if success {
            eprintln!("kick need and success.");
            best_solution.validate();
            assert!(best_solution.next(begin) == end || best_solution.prev(begin) == end);

            eprintln!(
                "kick eval = {}",
                evaluate(distance, &best_solution) as f64 * scale
            );
        }
    }

    best_solution = lkh::solve(
        distance,
        best_solution,
        LKHConfig {
            use_neighbor_cache: true,
            cache_filepath: get_cache_filepath(distance),
            debug: true,
            time_ms: 1000 * 60 * 60 * 1,
            start_kick_step: 30,
            kick_step_diff: 10,
            end_kick_step: 1000,
            fail_count_threashold: 50,
            max_depth: 7,
            neighbor_create_parallel: true,
            scale,
        },
    );
    eprintln!("finish lkh");
    eprintln!(
        "eval = {}",
        evaluate(distance, &best_solution) as f64 * scale
    );
    best_solution.save(&PathBuf::from_str("solution_all_lkh.tsp").unwrap());

    best_solution
}

struct PoseSimulator {
    pose_list: Vec<Pose>,
    // size64 を上下方向に動作させるか
    is_size64_vertical: bool,
}

impl PoseSimulator {
    fn new() -> PoseSimulator {
        PoseSimulator {
            pose_list: vec![Pose::new()],
            is_size64_vertical: false,
        }
    }

    fn raw_simulate_auto(&mut self, diff: DiffPose) {}

    fn raw_simulate(&mut self, to: (i16, i16)) {}
}

// subset に分けて、s-t パスを繋いで最適化する
// 適切な配置から開始すれば、128x128 矩形は割と自由に動けるという理由による
pub fn solve() {
    let mut orig_index_table = HashMap::<(i16, i16), u32>::new();
    {
        let mut index = 0u32;
        for y in (-128..=128).rev() {
            for x in -128..=128 {
                orig_index_table.insert((y, x), index);
                index += 1;
            }
        }
    }
    let orig_index_table = orig_index_table;

    // initialize
    // (0, 0) -> (64, 0)
    let mut initial_subpath = vec![];
    for y in 0..=64 {
        initial_subpath.push((y, 0));
    }

    // Upper Left
    // (64, 0) -> (0, -128)
    let ul_coord_list = extract_upper_left();
    let ul_subpath = calculate_subpath(
        ul_coord_list,
        (64, 0),
        (0, -128),
        &orig_index_table,
        "subpath_ul".to_string(),
        false,
    );

    // Lower Left
    // (0, -128) -> (-128, 0)
    let ll_coord_list = extract_lower_left();
    let ll_subpath = calculate_subpath(
        ll_coord_list,
        (0, -128),
        (-128, 0),
        &orig_index_table,
        "subpath_ll".to_string(),
        true,
    );

    // Lower Right
    // (-128, 0) -> (0, 128)
    let lr_coord_list = extract_lower_right();
    let lr_subpath = calculate_subpath(
        lr_coord_list,
        (-128, 0),
        (0, 128),
        &orig_index_table,
        "subpath_lr".to_string(),
        false,
    );

    // Upper Right
    // (0, 128) -> (64, 1)
    let ur_coord_list = extract_upper_right();
    let ur_subpath = calculate_subpath(
        ur_coord_list,
        (0, 128),
        (64, 1),
        &orig_index_table,
        "subpath_ur".to_string(),
        true,
    );

    // finalize
    // (64, 1) -> (1, 1) -> (0, 0)
    let mut final_subpath = vec![];
    for y in (1..=64).rev() {
        final_subpath.push((y, 1));
    }
    // 不要
    // final_subpath.push((0, 1));
    // final_subpath.push((0, 0));

    // merge solution

    let mut merged_solution = vec![];
    for subpath in [
        initial_subpath,
        ul_subpath,
        ll_subpath,
        lr_subpath,
        ur_subpath,
        final_subpath,
    ] {
        for coord in subpath.iter() {
            merged_solution.push(orig_index_table[coord]);
        }
        merged_solution.pop();
    }
    // 最後の final で push した分
    merged_solution.push(orig_index_table[&(1, 1)]);

    let final_solution = ArraySolution::from_array(merged_solution);
    final_solution.save(&PathBuf::from_str("final_solution.tsp").unwrap());
}
