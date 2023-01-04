use std::{path::PathBuf, str::FromStr, time::Instant};

use rand::{rngs::ThreadRng, Rng};

use crate::{
    array_solution::ArraySolution, bitset::BitSet, distance::DistanceFunction, evaluate::evaluate,
    intset::IntSet, neighbor_table::NeighborTable, segment_tree::SegmentTree, solution::Solution,
};

fn solve_inner<'a, T: Solution>(
    depth: usize,
    max_depth: usize,
    distance: &impl DistanceFunction,
    neighbor_table: &NeighborTable,
    current_flip: &mut SegmentTree<'a, T>,
    best_flip: &mut SegmentTree<'a, T>,
    edge_stack: &mut Vec<(u32, u32)>,
    gain: i64,
    best_gain: &mut i64,
    selected: &mut BitSet,
    rng: &mut ThreadRng,
) {
    if depth == max_depth {
        // 評価して最も良いゲインのものを保存
        if *best_gain < gain {
            *best_gain = gain;
            best_flip.copy_from(&current_flip);
        }
        return;
    }

    fn check<'a, T: Solution>(
        depth: usize,
        max_depth: usize,
        distance: &impl DistanceFunction,
        neighbor_table: &NeighborTable,
        current_flip: &mut SegmentTree<'a, T>,
        best_flip: &mut SegmentTree<'a, T>,
        edge_stack: &mut Vec<(u32, u32)>,
        gain: i64,
        best_gain: &mut i64,
        selected: &mut BitSet,
        f1: u32,
        t1: u32,
        f2: u32,
        t2: u32,
        rng: &mut ThreadRng,
    ) {
        if selected.test(f2) || selected.test(t2) {
            return;
        }
        selected.set(f2);
        selected.set(t2);
        current_flip.swap(t1, f2);

        let partial_gain = distance.distance(f1, t1) + distance.distance(f2, t2)
            - distance.distance(f1, f2)
            - distance.distance(t1, t2);

        // 新しくできた (f1, f2), (t1, t2) というエッジが次の交換対象
        // 2パターンあるのでどちらも探索
        for edge in [(f1, f2), (t1, t2)] {
            edge_stack.push(edge);
            solve_inner(
                depth + 1,
                max_depth,
                distance,
                neighbor_table,
                current_flip,
                best_flip,
                edge_stack,
                gain + partial_gain,
                best_gain,
                selected,
                rng,
            );
            edge_stack.pop();
        }

        current_flip.undo();
        selected.clear(f2);
        selected.clear(t2);
    }

    // edge stack のトップと入れ替える
    // from, to のどちらかに近い頂点を候補に入れたい
    let &(f1, t1) = edge_stack.last().unwrap();

    if rng.gen_bool(0.5) {
        for f2 in neighbor_table.neighbor_list(f1) {
            let t2 = current_flip.next(*f2);
            check(
                depth,
                max_depth,
                distance,
                neighbor_table,
                current_flip,
                best_flip,
                edge_stack,
                gain,
                best_gain,
                selected,
                f1,
                t1,
                *f2,
                t2,
                rng,
            );
        }
    } else {
        for t2 in neighbor_table.neighbor_list(t1) {
            let f2 = current_flip.prev(*t2);
            check(
                depth,
                max_depth,
                distance,
                neighbor_table,
                current_flip,
                best_flip,
                edge_stack,
                gain,
                best_gain,
                selected,
                f1,
                t1,
                f2,
                *t2,
                rng,
            );
        }
    }
}

pub struct LKHConfig {
    pub use_neighbor_cache: bool,
    pub cache_filepath: PathBuf,
    pub debug: bool,
    pub time_ms: u128,
    pub start_kick_step: usize,
    pub kick_step_diff: usize,
    pub end_kick_step: usize,
    pub fail_count_threashold: u32,
    pub max_depth: usize,
    pub neighbor_create_parallel: bool,
}

pub fn solve(
    distance: &(impl DistanceFunction + std::marker::Sync),
    mut solution: ArraySolution,
    config: LKHConfig,
) -> ArraySolution {
    let n = distance.dimension() as usize;
    // 解く

    let start = Instant::now();

    let neighbor_table = if config.use_neighbor_cache && config.cache_filepath.exists() {
        NeighborTable::load(&config.cache_filepath)
    } else {
        let table = NeighborTable::new(distance, 5, config.neighbor_create_parallel);
        if config.use_neighbor_cache {
            table.save(&config.cache_filepath);
        }
        table
    };

    let mut rng = rand::thread_rng();

    let mut dlb = IntSet::new(n);
    dlb.set_all();

    let mut eval = evaluate(distance, &solution);
    let mut selected = BitSet::new(n);

    let mut global_best_eval = eval;
    let mut global_best_solution = solution.clone();

    let mut no_random_step = config.start_kick_step;
    let mut no_continuous_fail_count = 0;

    for iter in 0.. {
        let a = dlb.random_select(&mut rng);

        selected.clear_all();

        let diff = {
            let mut current_tree = SegmentTree::new(&solution);
            let mut best_tree = SegmentTree::new(&solution);

            let mut best_gain = 0;

            let a_next = solution.next(a);
            let a_prev = solution.prev(a);

            let mut edge_stack = vec![];

            // iterative deeping
            for max_depth in 2..=config.max_depth {
                for (a, b) in [(a_prev, a), (a, a_next)] {
                    selected.set(a);
                    selected.set(b);
                    edge_stack.push((a, b));

                    solve_inner(
                        1,
                        max_depth,
                        distance,
                        &neighbor_table,
                        &mut current_tree,
                        &mut best_tree,
                        &mut edge_stack,
                        0,
                        &mut best_gain,
                        &mut selected,
                        &mut rng,
                    );

                    selected.clear(a);
                    selected.clear(b);
                    edge_stack.pop();
                }

                if best_gain > 0 {
                    break;
                }
            }
            if best_gain > 0 {
                Some((best_gain, best_tree.to_swap_list()))
            } else {
                None
            }
        };

        if let Some((gain, edge_list)) = diff {
            eval -= gain;
            for (from, to) in edge_list.into_iter() {
                solution.swap(from, to);
                dlb.push(from);
                dlb.push(to);
            }
        } else {
            dlb.remove(a);
        }

        if dlb.is_empty() {
            if config.debug {
                eprintln!("-----");
                eprintln!(
                    "step: {} (failcount: {})",
                    no_random_step, no_continuous_fail_count
                );
                eprintln!("iter: {}", iter);
                eprintln!("best eval: {}", eval as f64 / (255.0 * 10000.0));
                eprintln!("dlb size: {}", dlb.len());
            }

            if global_best_eval > eval {
                global_best_eval = eval;
                global_best_solution.copy_from(&solution);
                no_continuous_fail_count = 0;

                if config.debug {
                    global_best_solution.save(&PathBuf::from_str("solution_all_lkh.tsp").unwrap());
                }
            } else {
                solution.copy_from(&global_best_solution);
                no_continuous_fail_count += 1;
            }

            if no_continuous_fail_count == config.fail_count_threashold {
                no_random_step = (config.end_kick_step).min(no_random_step + config.kick_step_diff);
                no_continuous_fail_count = 0;
            }

            // random 2-opt kick
            // 近い部分のエッジを強制的に結ぶ kick
            // どうせ kick するなら、ある点の近傍をたくさん kick した方が変化させる意味があるから、
            // chain させる感じで変化をさせる。
            let mut a = rng.gen_range(0..n as u32);
            let mut b = solution.next(a);

            let mut selected = BitSet::new(n);
            selected.set(a);
            selected.set(b);

            'outer: for _step in 0..no_random_step {
                let mut counter = 0;
                while neighbor_table
                    .neighbor_list(a)
                    .iter()
                    .all(|v| selected.test(*v) || selected.test(solution.next(*v)))
                {
                    let a_size = neighbor_table.neighbor_list(a).len();
                    let a_idx = rng.gen_range(0..a_size);
                    a = neighbor_table.neighbor_list(a)[a_idx];

                    counter += 1;
                    if counter >= 30 {
                        break 'outer;
                    }
                }

                let c_size = neighbor_table.neighbor_list(a).len();
                let c_idx = rng.gen_range(0..c_size);
                let mut c = neighbor_table.neighbor_list(a)[c_idx];
                let mut d = solution.next(c);

                while selected.test(c) || selected.test(d) {
                    let c_idx = rng.gen_range(0..c_size);
                    c = neighbor_table.neighbor_list(a)[c_idx];
                    d = solution.next(c);
                }
                selected.set(c);
                selected.set(d);

                solution.swap(b, c);
                for id in [a, b, c, d] {
                    dlb.push(id);
                }

                (a, b) = (b, d);
            }
            eval = evaluate(distance, &solution);

            let end = Instant::now();
            if (end - start).as_millis() > config.time_ms {
                break;
            }
        }
    }
    global_best_solution
}
