use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    str::FromStr,
    time::Instant,
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
    util::{load_image, Cell, Coord, DiffPose, Direction, Pose},
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

    let filepath =
        PathBuf::from_str(format!("solution_split_lkh_{}.tsp", distance.name()).as_str()).unwrap();
    if filepath.exists() {
        return ArraySolution::load(&filepath);
    }

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

    let solutions = (0..96)
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

            (0, local_solution)
        })
        .collect::<Vec<_>>();

    for (_, mut local_solution) in solutions.into_iter() {
        // 並列化して最適化
        for _iter in 0..2 {
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

        if best_eval > eval {
            best_solution = local_solution;
            best_eval = eval;
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
            time_ms: 1000 * 60 * 5,
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
    color_list: Vec<Cell>,
    index_table: HashMap<(i16, i16), u32>,
}

impl PoseSimulator {
    fn new(color_list: Vec<Cell>) -> PoseSimulator {
        let mut index_table = HashMap::new();
        let mut index = 0;
        for y in (-128..=128 as i16).rev() {
            for x in -128..=128 as i16 {
                index_table.insert((y, x), index);
                index += 1;
            }
        }

        PoseSimulator {
            pose_list: vec![Pose::new()],
            color_list,
            index_table,
        }
    }

    fn current(&self) -> (i16, i16) {
        self.pose_list.last().unwrap().coord()
    }

    fn get_max_value(depth: usize) -> i16 {
        let ret = [64, 32, 16, 8, 4, 2, 1, 1];
        ret[depth]
    }

    fn create_coord_pose_map(
        &self,
        initial_pose: &Pose,
        is_64x64_vertical: bool,
    ) -> Vec<Vec<u128>> {
        // [(i16, i16)] -> Vec<u128>
        let mut coord_pose_map = vec![vec![]; 257 * 257];

        let encode_coord = |(y, x)| -> usize { ((y as i64 + 128) * 257 + x as i64 + 128) as usize };
        let key = encode_coord(initial_pose.coord());
        coord_pose_map[key].push(initial_pose.encode());

        let arm = initial_pose.arm_list;
        if is_64x64_vertical {
            for y_64 in -64..=64 {
                for x_32 in -32..=32 {
                    for x_16 in -16..=16 {
                        for x_8 in -8..=8 {
                            for x_4 in -4..=4 {
                                for x_2 in -2..=2 {
                                    for x_1_1 in -1..=1 {
                                        for x_1_2 in -1..=1 {
                                            let arm_list = [
                                                Coord::new(y_64, arm[0].x),
                                                Coord::new(arm[1].y, x_32),
                                                Coord::new(arm[2].y, x_16),
                                                Coord::new(arm[3].y, x_8),
                                                Coord::new(arm[4].y, x_4),
                                                Coord::new(arm[5].y, x_2),
                                                Coord::new(arm[6].y, x_1_1),
                                                Coord::new(arm[7].y, x_1_2),
                                            ];
                                            let pose = Pose { arm_list };
                                            let coord = pose.coord();
                                            coord_pose_map[encode_coord(coord)].push(pose.encode());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            for x_64 in -64..=64 {
                for y_32 in -32..=32 {
                    for y_16 in -16..=16 {
                        for y_8 in -8..=8 {
                            for y_4 in -4..=4 {
                                for y_2 in -2..=2 {
                                    for y_1_1 in -1..=1 {
                                        for y_1_2 in -1..=1 {
                                            let arm_list = [
                                                Coord::new(arm[0].y, x_64),
                                                Coord::new(y_32, arm[1].x),
                                                Coord::new(y_16, arm[2].x),
                                                Coord::new(y_8, arm[3].x),
                                                Coord::new(y_4, arm[4].x),
                                                Coord::new(y_2, arm[5].x),
                                                Coord::new(y_1_1, arm[6].x),
                                                Coord::new(y_1_2, arm[7].x),
                                            ];
                                            let pose = Pose { arm_list };
                                            let coord = pose.coord();
                                            coord_pose_map[encode_coord(coord)].push(pose.encode());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        coord_pose_map
    }

    fn simulate_manual(&mut self, diff: &DiffPose) {
        let mut last = self.pose_list.last().unwrap().clone();
        last.apply(diff);
        self.pose_list.push(last);
    }

    fn simulate_best_cost(
        &mut self,
        initial_pose: &Pose,
        point_seq: &Vec<(i16, i16)>,
        final_pose: &Pose,
        is_64x64_vertical: bool,
    ) {
        // 復元用の table
        let mut rev = HashMap::<u128, u128>::new();

        let mut cost_table = HashMap::<u128, i64>::new();

        // chokudai search
        let mut pose_buffer = vec![BinaryHeap::<(i64, u128)>::new(); point_seq.len()];
        pose_buffer[0].push((0, initial_pose.encode()));
        cost_table.insert(initial_pose.encode(), 0);

        let mut last_pose_set = HashMap::<u128, i64>::new();
        let mut best_cost = std::f64::MAX;

        let start = Instant::now();
        loop {
            for index in 0..point_seq.len() - 1 {
                let to = point_seq[index + 1];

                if let Some((cost, from_pose_enc)) = pose_buffer[index].pop() {
                    let cost = -cost;
                    let from_pose = Pose::decode(from_pose_enc);
                    for (to_pose_enc, to_cost) in self.cost(&from_pose, to, is_64x64_vertical) {
                        let next_cost = cost + to_cost;
                        if !cost_table.contains_key(&to_pose_enc)
                            || cost_table[&to_pose_enc] > next_cost
                        {
                            cost_table.insert(to_pose_enc, next_cost);
                            rev.insert(to_pose_enc, from_pose_enc);
                            pose_buffer[index + 1].push((-next_cost, to_pose_enc));

                            if index == point_seq.len() - 2 {
                                last_pose_set.insert(to_pose_enc, next_cost);
                            }
                        }
                    }
                }
            }

            for (k, v) in last_pose_set.iter() {
                let v = *v as f64 / (255.0 * 10000.0);
                if best_cost > v {
                    best_cost = v;
                    eprintln!("last cost: {}", v);
                }
            }

            let elapsed = (Instant::now() - start).as_millis();
            if elapsed > 30_000 {
                break;
            }
        }

        // 復元
        let mut pose_sequence = vec![];
        let mut cur = final_pose.encode();
        let init_enc = initial_pose.encode();
        while cur != init_enc {
            let prev_pose_enc = rev[&cur];
            let cur_pose = Pose::decode(cur);
            let prev_pose = Pose::decode(prev_pose_enc);
            let mut subseq = self.cost_pose_seq(&cur_pose, &prev_pose, is_64x64_vertical);
            // prev_pose を除外
            subseq.pop();
            pose_sequence.append(&mut subseq);
            cur = prev_pose_enc;
        }
        pose_sequence.reverse();
        self.pose_list.append(&mut pose_sequence);
    }

    // 整数で 255 倍されたリアルのコスト
    fn cost_one_step(&self, from: &Pose, to: &Pose) -> i64 {
        // 1手でたどり着く場合は迂回しないのが明らかに最善なので簡単
        let pose_cost = (from.cost(&to) * 255.0 * 10000.0) as i64;
        let from_color = self.color_list[self.index_table[&from.coord()] as usize];
        let to_color = self.color_list[self.index_table[&to.coord()] as usize];

        let dr = from_color.r.abs_diff(to_color.r) as i64;
        let dg = from_color.g.abs_diff(to_color.g) as i64;
        let db = from_color.b.abs_diff(to_color.b) as i64;

        let color_cost = (dr + dg + db) * 3 * 10000;

        pose_cost + color_cost
    }

    fn next_direction(
        &self,
        max_size: i16,
        coord: Coord,
        dy: i16,
        dx: i16,
        is_vertical: bool,
    ) -> Vec<Direction> {
        if is_vertical {
            let mut ret = vec![Direction::None];
            if dy > 0 && max_size > coord.y {
                ret.push(Direction::Up);
            }
            if dy < 0 && -max_size < coord.y {
                ret.push(Direction::Down);
            }
            ret
        } else {
            let mut ret = vec![Direction::None];
            if dx > 0 && max_size > coord.x {
                ret.push(Direction::Right);
            }
            if dx < 0 && -max_size < coord.x {
                ret.push(Direction::Left);
            }
            ret
        }
    }

    fn next_pose_list(
        &self,
        cur: &Pose,
        to_coord: (i16, i16),
        is_64x64_vertical: bool,
    ) -> Vec<Pose> {
        let cur_coord = cur.coord();
        let dx = to_coord.1 - cur_coord.1;
        let dy = to_coord.0 - cur_coord.0;

        let min_y = cur_coord.0.min(to_coord.0);
        let max_y = cur_coord.0.max(to_coord.0);
        let min_x = cur_coord.1.min(to_coord.1);
        let max_x = cur_coord.1.max(to_coord.1);

        let mut ret = vec![];
        for dir_64 in self.next_direction(64, cur.arm_list[0], dy, dx, is_64x64_vertical) {
            for dir_32 in self.next_direction(32, cur.arm_list[1], dy, dx, !is_64x64_vertical) {
                for dir_16 in self.next_direction(16, cur.arm_list[2], dy, dx, !is_64x64_vertical) {
                    for dir_8 in self.next_direction(8, cur.arm_list[3], dy, dx, !is_64x64_vertical)
                    {
                        for dir_4 in
                            self.next_direction(4, cur.arm_list[4], dy, dx, !is_64x64_vertical)
                        {
                            for dir_2 in
                                self.next_direction(2, cur.arm_list[5], dy, dx, !is_64x64_vertical)
                            {
                                for dir_1_1 in self.next_direction(
                                    1,
                                    cur.arm_list[6],
                                    dy,
                                    dx,
                                    !is_64x64_vertical,
                                ) {
                                    for dir_1_2 in self.next_direction(
                                        1,
                                        cur.arm_list[7],
                                        dy,
                                        dx,
                                        !is_64x64_vertical,
                                    ) {
                                        let diff_list = [
                                            dir_64, dir_32, dir_16, dir_8, dir_4, dir_2, dir_1_1,
                                            dir_1_2,
                                        ];
                                        let diff = DiffPose { diff_list };
                                        let mut next_pose = cur.clone();
                                        next_pose.apply(&diff);

                                        // from -> to へ遠回りせず行くためのルートを計算するので、from -> to 内の区間で十分
                                        let next_coord = next_pose.coord();
                                        if min_y <= next_coord.0
                                            && next_coord.0 <= max_y
                                            && min_x <= next_coord.1
                                            && next_coord.1 <= max_x
                                        {
                                            ret.push(next_pose);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        ret
    }

    // 正確なコスト計算
    // from -> to へ行く過程で、ロスなく迎えるコストを全計算する
    fn cost_pose_seq(&self, from: &Pose, to: &Pose, is_64x64_vertical: bool) -> Vec<Pose> {
        let mut que = BinaryHeap::new();
        let mut cost_table = HashMap::<u128, i64>::new();

        let from_enc = from.encode();
        let to_enc = to.encode();
        let to_coord = to.coord();
        let mut rev = HashMap::<u128, u128>::new();

        que.push((0, from_enc));
        cost_table.insert(from_enc, 0);

        while let Some((cost, enc)) = que.pop() {
            let cost = -cost;
            if cost_table[&enc] < cost {
                continue;
            }

            if enc == to_enc {
                // 復元
                let mut ret = vec![];
                let mut cur = to_enc;
                while cur != from_enc {
                    ret.push(Pose::decode(cur));
                    cur = rev[&cur];
                }
                ret.push(Pose::decode(cur));
                ret.reverse();
                return ret;
            }

            // 1歩で行ける全てのコストを計上
            let pose = Pose::decode(enc);

            for next_pose in self.next_pose_list(&pose, to_coord, is_64x64_vertical) {
                let next_enc = next_pose.encode();
                let diff_cost = self.cost_one_step(&pose, &next_pose);
                let next_cost = cost + diff_cost;

                if !cost_table.contains_key(&next_enc) || cost_table[&next_enc] > next_cost {
                    rev.insert(next_enc, enc);
                    cost_table.insert(next_enc, next_cost);
                    que.push((-next_cost, next_enc));
                }
            }
        }
        unreachable!();
    }

    // 正確なコスト計算
    // from -> to へ行く過程で、ロスなく迎えるコストを全計算する
    fn cost(
        &self,
        from: &Pose,
        to_coord: (i16, i16),
        is_64x64_vertical: bool,
    ) -> HashMap<u128, i64> {
        let mut ret = HashMap::new();

        let mut que = BinaryHeap::new();
        let mut cost_table = HashMap::<u128, i64>::new();

        let from_enc = from.encode();
        que.push((0, from_enc));
        cost_table.insert(from_enc, 0);

        while let Some((cost, enc)) = que.pop() {
            let cost = -cost;
            if cost_table[&enc] < cost {
                continue;
            }

            // 1歩で行ける全てのコストを計上
            let pose = Pose::decode(enc);

            if pose.coord() == to_coord {
                ret.insert(enc, cost);
            }

            for next_pose in self.next_pose_list(&pose, to_coord, is_64x64_vertical) {
                let next_enc = next_pose.encode();
                let diff_cost = self.cost_one_step(&pose, &next_pose);
                let next_cost = cost + diff_cost;

                if !cost_table.contains_key(&next_enc) || cost_table[&next_enc] > next_cost {
                    cost_table.insert(next_enc, next_cost);
                    que.push((-next_cost, next_enc));
                }
            }
        }
        ret
    }

    fn save(&self, filepath: &PathBuf) {
        let f = File::create(filepath).unwrap();
        let mut writer = BufWriter::new(f);
        writer.write("configuration\n".as_bytes()).unwrap();
        for pose in self.pose_list.iter() {
            writer.write(pose.to_string().as_bytes()).unwrap();
        }
    }
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
    let color_table = load_image(&PathBuf::from_str("data/image.csv").unwrap());
    let mut pose_simulator = PoseSimulator::new(color_table.clone());

    // initialize
    for _iter in 0..64 {
        let mut diff = DiffPose {
            diff_list: [Direction::None; 8],
        };
        diff.diff_list[0] = Direction::Up;
        pose_simulator.simulate_manual(&diff);
    }

    let intermediate = vec![
        [
            // S'
            (64, 64),
            (0, -32),
            (0, -16),
            (0, -8),
            (0, -4),
            (0, -2),
            (0, -1),
            (0, -1),
        ],
        [
            // B
            (64, -64),
            (-32, -32),
            (-16, -16),
            (-8, -8),
            (-4, -4),
            (-2, -2),
            (-1, -1),
            (-1, -1),
        ],
        [
            // C
            (-64, -64),
            (-32, 32),
            (-16, 16),
            (-8, 8),
            (-4, 4),
            (-2, 2),
            (-1, 1),
            (-1, 1),
        ],
        [
            // D
            (-64, 64),
            (32, 32),
            (16, 16),
            (8, 8),
            (4, 4),
            (2, 2),
            (1, 1),
            (1, 1),
        ],
        [
            // E
            (0, 64),
            (32, -32),
            (16, -16),
            (8, -8),
            (4, -4),
            (2, -2),
            (1, -1),
            (1, 0),
        ],
    ];
    let intermediate = intermediate
        .into_iter()
        .map(|array| array.map(|(y, x)| Coord::new(y, x)))
        .map(|arm_list| Pose { arm_list })
        .collect::<Vec<_>>();

    let subpath_list = vec![
        ul_subpath.clone(),
        ll_subpath.clone(),
        lr_subpath.clone(),
        ur_subpath.clone(),
    ];
    for index in 0..4 {
        let initial_pose = intermediate[index].clone();
        let final_pose = intermediate[index + 1].clone();
        assert_eq!(*subpath_list[index].first().unwrap(), initial_pose.coord());
        assert_eq!(*subpath_list[index].last().unwrap(), final_pose.coord());
        let is_64x64_vertical = index % 2 == 1;
        pose_simulator.simulate_best_cost(
            &initial_pose,
            &subpath_list[index],
            &final_pose,
            is_64x64_vertical,
        );
    }
    for i in 1..7 {
        let max_size_list = [64, 32, 16, 8, 4, 2, 1, 1];
        for _iter in 0..max_size_list[i] {
            let mut diff = DiffPose {
                diff_list: [Direction::None; 8],
            };
            diff.diff_list[i] = Direction::Down;
            pose_simulator.simulate_manual(&diff);
        }
    }
    {
        let mut diff = DiffPose {
            diff_list: [Direction::None; 8],
        };
        diff.diff_list[7] = Direction::Left;
        pose_simulator.simulate_manual(&diff);
    }
    {
        let mut diff = DiffPose {
            diff_list: [Direction::None; 8],
        };
        diff.diff_list[7] = Direction::Down;
        pose_simulator.simulate_manual(&diff);
    }
    pose_simulator.save(&PathBuf::from_str("final_solution.csv").unwrap());

    // merge solutionp
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

    // eval
    let mut eval = 0.0;
    for i in 0..pose_simulator.pose_list.len() - 1 {
        let pose = pose_simulator.pose_list[i];
        let index = orig_index_table[&pose.coord()] as usize;
        let next_pose = pose_simulator.pose_list[i + 1];
        let next_index = orig_index_table[&next_pose.coord()] as usize;

        let pos_cost = pose.cost(&next_pose);

        let dr = color_table[index].r.abs_diff(color_table[next_index].r);
        let dg = color_table[index].g.abs_diff(color_table[next_index].g);
        let db = color_table[index].b.abs_diff(color_table[next_index].b);
        let color_cost = (dr + dg + db) as f64 * 3.0 / 255.0;
        eval += pos_cost + color_cost;
    }
    eprintln!("final eval: {}", eval);
}
