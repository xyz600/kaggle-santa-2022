use std::path::PathBuf;

use crate::{
    array_solution::ArraySolution, distance::DistanceFunction, evaluate::evaluate, intset::IntSet,
    neighbor_table::NeighborTable, solution::Solution,
    two_level_tree_solution::TwoLeveltreeSolution,
};

pub struct Opt2Config {
    pub use_neighbor_cache: bool,
    pub cache_filepath: PathBuf,
    pub debug: bool,
    pub neighbor_create_parallel: bool,
    pub scale: f64,
}

pub fn solve(
    distance: &(impl DistanceFunction + std::marker::Sync),
    solution: ArraySolution,
    config: Opt2Config,
) -> ArraySolution {
    let n = solution.len();

    let mut tlt = TwoLeveltreeSolution::<1000>::new(&solution);

    let neighbor_table = if config.use_neighbor_cache && config.cache_filepath.exists() {
        NeighborTable::load(&config.cache_filepath)
    } else {
        let table = NeighborTable::new(distance, 4, config.neighbor_create_parallel);
        if config.use_neighbor_cache {
            table.save(&config.cache_filepath);
        }
        table
    };

    let mut rng = rand::thread_rng();

    let mut dlb = IntSet::new(n);
    dlb.set_all();

    let mut eval = evaluate(distance, &tlt);

    for iter in 0.. {
        let a = dlb.random_select(&mut rng);
        let b = tlt.next(a);

        let mut best_gain = 0;
        let mut best_c = 0;

        for c in neighbor_table.neighbor_list(a) {
            let c = *c;
            let d = tlt.next(c);
            if a == d || b == c || b == d {
                continue;
            }

            let gain = distance.distance(a, b) + distance.distance(c, d)
                - distance.distance(a, c)
                - distance.distance(b, d);
            if gain > best_gain {
                best_gain = gain;
                best_c = c;
            }
        }
        if best_gain > 0 {
            dlb.push(a);
            dlb.push(b);
            dlb.push(best_c);
            let best_d = tlt.next(best_c);
            dlb.push(best_d);

            tlt.swap(b, best_c);
            eval -= best_gain;
        } else {
            dlb.remove(a);
        }

        if config.debug && (iter % n == 0 || dlb.is_empty()) {
            eprintln!("iter = {}, eval = {}", iter, eval as f64 * config.scale);
            eprintln!("dlb size = {}", dlb.len());
        }
        if dlb.is_empty() {
            break;
        }
    }
    tlt.to_array_solution()
}
