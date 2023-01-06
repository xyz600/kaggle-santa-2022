use std::{path::PathBuf, str::FromStr};

use lib::{
    array_solution::ArraySolution,
    distance::DistanceFunction,
    divide_and_conqure_solver::{self, DivideAndConqureConfig},
    evaluate::evaluate,
    lkh::{self, LKHConfig},
    opt3::{self, Opt3Config},
    solution::Solution,
};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

fn get_cache_filepath(distance: &impl DistanceFunction) -> PathBuf {
    PathBuf::from_str(format!("{}.cache", distance.name()).as_str()).unwrap()
}

pub fn solve_tsp(
    distance: &(impl DistanceFunction + std::marker::Sync),
    init_solution: ArraySolution,
    scale: f64,
) -> ArraySolution {
    let mut solution = ArraySolution::new(init_solution.len());

    {
        // cache 作成用
        opt3::solve(
            distance,
            init_solution.clone(),
            Opt3Config {
                use_neighbor_cache: true,
                cache_filepath: get_cache_filepath(distance),
                debug: false,
                neighbor_create_parallel: true,
                scale: scale,
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
                        scale: scale,
                    },
                );

                let mut id = 0;
                let mut longest_edge = 0;
                for _iter in 0..local_solution.len() {
                    let next_id = local_solution.next(id);
                    let edge = distance.distance(id, next_id);
                    longest_edge = longest_edge.max(edge);
                    id = next_id;
                }
                eprintln!("longest edge: {}", longest_edge as f64 * scale);
                (longest_edge, local_solution)
            })
            .collect::<Vec<_>>();

        // 最長辺が小さい初期解を頑張って引く
        let mut best_longest_edge = std::i64::MAX;
        for (longest_edge, local_solution) in solutions.into_iter() {
            if best_longest_edge > longest_edge {
                best_longest_edge = longest_edge;
                solution = local_solution;
            }
        }
    }
    solution.save(&PathBuf::from_str("solution_opt3.tsp").unwrap());

    let mut solution = lkh::solve(
        distance,
        solution,
        LKHConfig {
            use_neighbor_cache: true,
            cache_filepath: get_cache_filepath(distance),
            debug: true,
            time_ms: 60_000,
            start_kick_step: 30,
            kick_step_diff: 10,
            end_kick_step: distance.dimension() as usize / 10,
            fail_count_threashold: 50,
            max_depth: 6,
            neighbor_create_parallel: true,
            scale: scale,
        },
    );
    eprintln!("finish initial lkh.");
    eprintln!("eval = {}", evaluate(distance, &solution) as f64 * scale);
    solution.save(&PathBuf::from_str("solution_initial_lkh.tsp").unwrap());

    // 分割して並列化
    let mut start_kick_step = 30;
    let mut time_ms = 30_000;
    let mut best_eval = evaluate(distance, &solution);

    for iter in 1.. {
        solution = divide_and_conqure_solver::solve(
            distance,
            &solution,
            DivideAndConqureConfig {
                no_split: 12,
                debug: false,
                time_ms,
                start_kick_step,
                kick_step_diff: 10,
                end_kick_step: distance.dimension() as usize / 10,
                fail_count_threashold: 50,
                max_depth: 7,
                scale: scale,
            },
        );
        let eval = evaluate(distance, &solution);
        eprintln!("finish splited lkh {} times.", iter);
        eprintln!("eval = {}", eval as f64 * scale);
        if best_eval == eval {
            start_kick_step += 10;
            time_ms += 30_000;
        } else {
            solution.save(&PathBuf::from_str("solution_split_lkh.tsp").unwrap());
        }
        best_eval = eval;

        if start_kick_step == 100 {
            break;
        }
    }

    let solution = lkh::solve(
        distance,
        solution,
        LKHConfig {
            use_neighbor_cache: true,
            cache_filepath: get_cache_filepath(distance),
            debug: true,
            time_ms: 1000 * 60 * 60 * 12,
            start_kick_step: 30,
            kick_step_diff: 10,
            end_kick_step: 1000,
            fail_count_threashold: 50,
            max_depth: 7,
            neighbor_create_parallel: true,
            scale: scale,
        },
    );
    eprintln!("finish lkh");
    eprintln!("eval = {}", evaluate(distance, &solution) as f64 * scale);
    solution.save(&PathBuf::from_str("solution_all_lkh.tsp").unwrap());

    solution
}
