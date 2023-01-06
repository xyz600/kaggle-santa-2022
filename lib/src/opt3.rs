use std::path::PathBuf;

use crate::{
    array_solution::ArraySolution, bitset::BitSet, distance::DistanceFunction, evaluate::evaluate,
    intset::IntSet, neighbor_table::NeighborTable, solution::Solution,
    two_level_tree_solution::TwoLeveltreeSolution,
};

#[derive(Debug)]
enum NeighborPattern {
    None,
    Pat1((u32, u32)),
    Pat2((u32, u32), (u32, u32)),
    Pat3((u32, u32), (u32, u32), (u32, u32)),
}

pub struct Opt3Config {
    pub use_neighbor_cache: bool,
    pub cache_filepath: PathBuf,
    pub debug: bool,
    pub neighbor_create_parallel: bool,
    pub scale: f64,
}

// https://en.wikipedia.org/wiki/3-opt
pub fn solve(
    distance: &(impl DistanceFunction + std::marker::Sync),
    solution: ArraySolution,
    config: Opt3Config,
) -> ArraySolution {
    let n = solution.len();

    let mut tlt = TwoLeveltreeSolution::<1000>::new(&solution);

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

    let mut eval = evaluate(distance, &tlt);
    let mut selected = BitSet::new(n);

    for iter in 0.. {
        let a = dlb.random_select(&mut rng);

        selected.clear_all();

        let mut best_gain = 0;
        let mut best_pat = NeighborPattern::None;

        let a_next = tlt.next(a);
        let a_prev = tlt.prev(a);

        for (a, b) in [(a_prev, a), (a, a_next)] {
            selected.set(a);
            selected.set(b);

            // a-c をくっつけるとよさそう
            for c in neighbor_table.neighbor_list(a) {
                let c_next = tlt.next(*c);
                let c_prev = tlt.prev(*c);

                for (c, d) in [(c_prev, *c), (*c, c_next)] {
                    if selected.test(c) || selected.test(d) {
                        continue;
                    }
                    selected.set(c);
                    selected.set(d);

                    for e in neighbor_table.neighbor_list(c) {
                        let e_next = tlt.next(*e);
                        let e_prev = tlt.prev(*e);

                        for (e, f) in [(e_prev, *e), (*e, e_next)] {
                            if selected.test(e) || selected.test(f) {
                                continue;
                            }
                            selected.set(e);
                            selected.set(f);

                            let dist = |i1, i2| distance.distance(i1, i2);

                            let (ca, cb, cc, cd, ce, cf) = if tlt.between(c, a, e) {
                                (a, b, c, d, e, f)
                            } else {
                                (a, b, e, f, c, d)
                            };

                            // case 1
                            // [(a, b), (c, d)] -> [(a, c), (b, d)]
                            let gain1 = dist(ca, cb) + dist(cc, cd) - dist(ca, cc) - dist(cb, cd);
                            if gain1 > best_gain {
                                best_gain = gain1;
                                best_pat = NeighborPattern::Pat1((cb, cc))
                            }

                            // case 2
                            // [(c, d), (e, f)] -> [(c, e), (d, f)]
                            let gain2 = dist(cc, cd) + dist(ce, cf) - dist(cc, ce) - dist(cd, cf);
                            if gain2 > best_gain {
                                best_gain = gain2;
                                best_pat = NeighborPattern::Pat1((cd, ce));
                            }

                            // case 4
                            // [(a, b), (c, d), (e, f)] -> [(a, c), (b, e), (d, f)]
                            // [(a, b), (c, d), (e, f)] -> [(a, c), (b, d), (e, f)] -> [(a, c), (b, e), (d, f)]
                            let gain4 = dist(ca, cb) + dist(cc, cd) + dist(ce, cf)
                                - dist(ca, cc)
                                - dist(cb, ce)
                                - dist(cd, cf);
                            if gain4 > best_gain {
                                best_gain = gain4;
                                best_pat = NeighborPattern::Pat2((cb, cc), (cd, ce));
                            }

                            // case 6
                            // [(a, b), (c, d), (e, f)] -> [(a, d), (e, c), (b, f)]
                            // [(a, b), (c, d), (e, f)] -> [(a, e), (d, c), (b, f)] -> [(a, d), (e, c), (b, f)]
                            let gain6 = dist(ca, cb) + dist(cc, cd) + dist(ce, cf)
                                - dist(ca, cd)
                                - dist(ce, cc)
                                - dist(cb, cf);
                            if gain6 > best_gain {
                                best_gain = gain6;
                                best_pat = NeighborPattern::Pat2((cb, ce), (ce, cd));
                            }

                            // case 7
                            // [(a, b), (c, d), (e, f)] -> [(a, d), (e, b), (c, f)]
                            // [(a, b), (c, d), (e, f)] -> [(a, e), (d, c), (b, f)] -> [(a, d), (e, c), (b, f)] -> [(a, d), (e, b), (c, f)]
                            let gain7 = dist(ca, cb) + dist(cc, cd) + dist(ce, cf)
                                - dist(ca, cd)
                                - dist(ce, cb)
                                - dist(cc, cf);
                            if gain7 > best_gain {
                                best_gain = gain7;
                                best_pat = NeighborPattern::Pat3((cb, ce), (ce, cd), (cc, cb));
                            }

                            selected.clear(e);
                            selected.clear(f);
                        }
                    }

                    selected.clear(c);
                    selected.clear(d);
                }
            }

            // a-e をくっつけるとよさそう
            for e in neighbor_table.neighbor_list(a) {
                let e_next = tlt.next(*e);
                let e_prev = tlt.prev(*e);

                for (e, f) in [(e_prev, *e), (*e, e_next)] {
                    if selected.test(e) || selected.test(f) {
                        continue;
                    }
                    selected.set(e);
                    selected.set(f);

                    for c in neighbor_table.neighbor_list(f) {
                        let c_next = tlt.next(*c);
                        let c_prev = tlt.prev(*c);

                        for (c, d) in [(c_prev, *c), (*c, c_next)] {
                            if selected.test(c) || selected.test(d) {
                                continue;
                            }
                            selected.set(c);
                            selected.set(d);

                            let dist = |i1, i2| distance.distance(i1, i2);

                            let (ca, cb, cc, cd, ce, cf) = if tlt.between(c, a, e) {
                                (a, b, c, d, e, f)
                            } else {
                                (a, b, e, f, c, d)
                            };

                            // case 3
                            // [(a, b), (e, f)] -> [(a, e), (b, f)]
                            let gain3 = dist(ca, cb) + dist(ce, cf) - dist(ca, ce) - dist(cb, cf);
                            if gain3 > best_gain {
                                best_gain = gain3;
                                best_pat = NeighborPattern::Pat1((cf, ca));
                            }

                            // case 5
                            // [(a, b), (c, d), (e, f)] -> [(a, e), (d, b), (c, f)]
                            // [(a, b), (c, d), (e, f)] -> [(a, e), (d, c), (b, f)] -> [(a, e), (d, b), (c, f)]
                            let gain5 = dist(ca, cb) + dist(cc, cd) + dist(ce, cf)
                                - dist(ca, ce)
                                - dist(cd, cb)
                                - dist(cc, cf);
                            if gain5 > best_gain {
                                best_gain = gain5;
                                best_pat = NeighborPattern::Pat2((cb, ce), (cc, cb));
                            }
                            selected.clear(c);
                            selected.clear(d);
                        }
                    }
                    selected.clear(e);
                    selected.clear(f);
                }
            }

            selected.clear(a);
            selected.clear(b);
        }

        // swap
        match best_pat {
            NeighborPattern::None => {
                dlb.remove(a);
            }
            NeighborPattern::Pat1((i1, i2)) => {
                tlt.swap(i1, i2);
                for i in [i1, i2] {
                    dlb.push(i);
                }
                eval -= best_gain;
            }
            NeighborPattern::Pat2((i1, i2), (i3, i4)) => {
                tlt.swap(i1, i2);
                tlt.swap(i3, i4);
                for i in [i1, i2, i3, i4] {
                    dlb.push(i);
                }
                eval -= best_gain;
            }
            NeighborPattern::Pat3((i1, i2), (i3, i4), (i5, i6)) => {
                tlt.swap(i1, i2);
                tlt.swap(i3, i4);
                tlt.swap(i5, i6);
                for i in [i1, i2, i3, i4, i5, i6] {
                    dlb.push(i);
                }
                eval -= best_gain;
            }
        }

        if config.debug && (iter % (n / 10) == 0 || dlb.is_empty()) {
            eprintln!("iter = {}, eval = {}", iter, eval as f64 * config.scale);
            eprintln!("dlb size = {}", dlb.len());
        }
        if dlb.is_empty() {
            break;
        }
    }
    tlt.to_array_solution()
}
