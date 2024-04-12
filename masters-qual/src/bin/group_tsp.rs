#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::neg_multiply)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, VecDeque},
};

use itertools::Itertools;
use proconio::{fastout, input, marker::Chars};
use rand::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

fn main() {
    let start = std::time::Instant::now();
    let time_limit = 1.8;
    let time_keeper = TimeKeeper::new(time_limit);
    let mut rng = rand_pcg::Pcg64Mcg::new(12345);
    let input = read_input();
    let target = make_target(&input, &mut rng, &time_keeper);
    let mut state = State::new(&input, &target);
    state.quick_sort_by_group(&input, &target);
    state.output();
    let cost = calc_cost(&state.board, &input);
    let score = calc_score(input.cost, cost);
    eprintln!("Cost: {}", cost);
    eprintln!("Score: {}", score);

    #[allow(unused_mut, unused_assignments)]
    let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
    #[cfg(feature = "local")]
    {
        eprintln!("Local Mode");
        elapsed_time *= 0.55;
    }
    eprintln!("Elapsed: {}", (elapsed_time * 1000.0) as usize);
}

struct Target {
    board: DynamicMap2d<i64>,
    start: Coord,
    goal: Coord,
    group_num: usize,
    group_board: DynamicMap2d<usize>,
    group_dist: Vec<Vec<usize>>,
    num_to_group: Vec<usize>,
}

fn make_target(input: &Input, rng: &mut rand_pcg::Pcg64Mcg, time_keeper: &TimeKeeper) -> Target {
    let (init_board, start, goal) = make_init_board_head_and_tail_start(input);
    let target_board = annealing(input, init_board, rng, time_keeper);
    let depth = if input.n <= 15 {
        2
    } else if input.n <= 30 {
        3
    } else {
        4
    };
    let group_num = 1_usize << depth;
    let (group_board, group_dist) = make_group_from_target_board(depth, input, rng, &target_board);
    // let (group_board, group_dist) = make_group_random(group_num, input, rng, time_keeper);
    let mut num_to_group = vec![!0; input.n2];
    for i in 0..input.n {
        for j in 0..input.n {
            let pos = Coord::new(i, j);
            let num = target_board[pos] - 1;
            num_to_group[num as usize] = group_board[pos];
        }
    }
    Target {
        board: target_board,
        start,
        goal,
        group_num,
        group_board,
        group_dist,
        num_to_group,
    }
}

fn make_group_from_target_board(
    depth: usize,
    input: &Input,
    rng: &mut rand_pcg::Pcg64Mcg,
    board: &DynamicMap2d<i64>,
) -> (DynamicMap2d<usize>, Vec<Vec<usize>>) {
    let mut Q = VecDeque::new();
    let mut sep = vec![];
    Q.push_back((0, 0, input.n2));
    while let Some((d, left, right)) = Q.pop_front() {
        if d == depth {
            sep.push((left, right));
            continue;
        }
        let mid = (left + right) / 2;
        Q.push_back((d + 1, left, mid));
        Q.push_back((d + 1, mid, right));
    }
    sep.sort();

    let mut group = DynamicMap2d::new_with(!0, input.n);
    for i in 0..input.n {
        for j in 0..input.n {
            let pos = Coord::new(i, j);
            let x = (board[pos] - 1) as usize;
            for (g, &(left, right)) in sep.iter().enumerate() {
                if left <= x && x < right {
                    group[pos] = g;
                }
            }
        }
    }

    let group_num = 1_usize << depth;
    let mut used = DynamicMap2d::new_with(false, input.n);
    let mut connected = vec![vec![]; group_num];
    for i in 0..input.n {
        for j in 0..input.n {
            let pos = Coord::new(i, j);
            if used[pos] {
                continue;
            }
            let g = group[pos];
            let mut v = vec![pos];
            let mut Q = VecDeque::new();
            used[pos] = true;
            Q.push_back(pos);
            while let Some(pos) = Q.pop_front() {
                for (_, nxt) in input.legal_actions[pos].iter() {
                    if used[*nxt] {
                        continue;
                    }
                    if group[*nxt] != g {
                        continue;
                    }
                    used[*nxt] = true;
                    Q.push_back(*nxt);
                    v.push(*nxt);
                }
            }
            if connected[g].len() < v.len() {
                connected[g] = v;
            }
        }
    }

    let mut modified_group = DynamicMap2d::new_with(!0, input.n);
    for g in 0..group_num {
        for pos in connected[g].iter() {
            modified_group[*pos] = g;
        }
    }

    let mut counts = vec![0; group_num];
    for (i, v) in connected.iter().enumerate() {
        counts[i] = v.len();
    }

    let mut not_used = vec![];
    for i in 0..input.n {
        for j in 0..input.n {
            let pos = Coord::new(i, j);
            if modified_group[pos] == !0 {
                not_used.push(pos);
            }
        }
    }
    not_used.shuffle(rng);

    let INF = 1_usize << 60;
    while let Some(pos) = not_used.pop() {
        let mut cnt = INF;
        let mut g = 0;
        for (_, nxt) in input.legal_actions[pos].iter() {
            if modified_group[*nxt] == !0 {
                continue;
            }
            if counts[modified_group[*nxt]] < cnt {
                cnt = counts[modified_group[*nxt]];
                g = modified_group[*nxt];
            }
        }
        if cnt == INF {
            not_used.push(pos);
            not_used.shuffle(rng);
        } else {
            modified_group[pos] = g;
        }
    }

    #[cfg(feature = "local")]
    visualizer::vis_grp(
        input.n,
        &input.vs,
        &input.hs,
        &modified_group.to_2d_vec(),
        group_num,
    );

    let mut G = vec![vec![]; group_num];
    for i in 0..input.n {
        for j in 0..input.n {
            let pos = Coord::new(i, j);
            for (_, nxt) in input.legal_actions[pos].iter() {
                if modified_group[pos] != modified_group[*nxt] {
                    G[modified_group[pos]].push(modified_group[*nxt]);
                    G[modified_group[*nxt]].push(modified_group[pos]);
                }
            }
        }
    }
    for g in 0..group_num {
        G[g].sort();
        G[g].dedup();
    }

    let mut dist = vec![vec![INF; group_num]; group_num];
    for g in 0..group_num {
        let mut Q = VecDeque::new();
        dist[g][g] = 0;
        Q.push_back(g);
        while let Some(pos) = Q.pop_front() {
            for nxt in &G[pos] {
                if dist[g][pos] + 1 < dist[g][*nxt] {
                    dist[g][*nxt] = dist[g][pos] + 1;
                    Q.push_back(*nxt);
                }
            }
        }
    }

    (modified_group, dist)
}

fn make_group_random(
    group_num: usize,
    input: &Input,
    rng: &mut rand_pcg::Pcg64Mcg,
    time_keeper: &TimeKeeper,
) -> (DynamicMap2d<usize>, Vec<Vec<usize>>) {
    let mut best_group = DynamicMap2d::new_with(!0, input.n);
    let INF = 1_usize << 60;
    let mut best_score = INF;

    while time_keeper.get_time() < 3.0 {
        let mut centers = FxHashSet::default();
        while centers.len() < group_num {
            let i = rng.gen_range(0..input.n);
            let j = rng.gen_range(0..input.n);
            centers.insert((i, j));
        }
        let centers = centers
            .into_iter()
            .map(|(i, j)| Coord::new(i, j))
            .collect_vec();

        let mut group = DynamicMap2d::new_with(!0, input.n);
        let mut Qs = vec![VecDeque::new(); group_num];
        let mut cnt = 0;
        for g in 0..group_num {
            let pos = centers[g];
            Qs[g].push_back(pos);
            group[pos] = g;
            cnt += 1;
        }

        'a: loop {
            for g in 0..group_num {
                if Qs[g].is_empty() {
                    continue;
                }
                let pos = Qs[g].pop_front().unwrap();
                for (_, nxt) in input.legal_actions[pos].iter() {
                    if group[*nxt] != !0 {
                        continue;
                    }
                    group[*nxt] = g;
                    cnt += 1;
                    Qs[g].push_back(*nxt);
                    if cnt == input.n2 {
                        break 'a;
                    }
                }
            }
        }

        let mut coords = vec![vec![]; group_num];
        for i in 0..input.n {
            for j in 0..input.n {
                let pos = Coord::new(i, j);
                coords[group[pos]].push((i, j));
            }
        }
        let mut score = 0;
        for g in 0..group_num {
            let mut row_max = 0;
            let mut row_min = INF;
            let mut col_max = 0;
            let mut col_min = INF;
            for &(row, col) in coords[g].iter() {
                row_max = row_max.max(row);
                col_max = col_max.max(col);
                row_min = row_min.min(row);
                col_min = col_min.min(col);
            }
            score = score.max(row_max - row_min + col_max - col_min);
        }

        let mut counts = vec![0; group_num];
        for i in 0..input.n {
            for j in 0..input.n {
                let pos = Coord::new(i, j);
                counts[group[pos]] += 1;
            }
        }
        let mx = counts.iter().max().unwrap();
        let mn = counts.iter().min().unwrap();
        score += mx - mn;

        if score < best_score {
            best_score = score;
            best_group = group;
        }
    }
    #[cfg(feature = "local")]
    visualizer::vis_grp(
        input.n,
        &input.vs,
        &input.hs,
        &best_group.to_2d_vec(),
        group_num,
    );

    let mut G = vec![vec![]; group_num];
    for i in 0..input.n {
        for j in 0..input.n {
            let pos = Coord::new(i, j);
            for (_, nxt) in input.legal_actions[pos].iter() {
                if best_group[pos] != best_group[*nxt] {
                    G[best_group[pos]].push(best_group[*nxt]);
                    G[best_group[*nxt]].push(best_group[pos]);
                }
            }
        }
    }
    for g in 0..group_num {
        G[g].sort();
        G[g].dedup();
    }

    let mut dist = vec![vec![INF; group_num]; group_num];
    for g in 0..group_num {
        let mut Q = VecDeque::new();
        dist[g][g] = 0;
        Q.push_back(g);
        while let Some(pos) = Q.pop_front() {
            for nxt in &G[pos] {
                if dist[g][pos] + 1 < dist[g][*nxt] {
                    dist[g][*nxt] = dist[g][pos] + 1;
                    Q.push_back(*nxt);
                }
            }
        }
    }

    (best_group, dist)
}

fn make_init_board_head_and_tail_start(input: &Input) -> (DynamicMap2d<i64>, Coord, Coord) {
    let mut cands = vec![];
    for i in 0..input.n {
        for j in 0..input.n {
            let pos = Coord::new(i, j);
            cands.push((input.legal_actions[pos].len(), i, j));
        }
    }
    cands.sort();
    let (_, i, j) = cands[0];
    let start = Coord::new(i, j);

    let INF = 1_usize << 60;
    let mut dist = DynamicMap2d::new_with(INF, input.n);
    let mut Q = VecDeque::new();

    dist[start] = 0;
    Q.push_back(start);
    while let Some(pos) = Q.pop_front() {
        for (_, nxt) in input.legal_actions[pos].iter() {
            if dist[pos] + 1 < dist[*nxt] {
                dist[*nxt] = dist[pos] + 1;
                Q.push_back(*nxt);
            }
        }
    }

    let mut cands = vec![];
    for i in 0..input.n {
        for j in 0..input.n {
            let pos = Coord::new(i, j);
            cands.push((dist[pos], Reverse(input.legal_actions[pos].len()), i, j));
        }
    }
    cands.sort();
    cands.reverse();
    let (_, _, i, j) = cands[0];
    let goal = Coord::new(i, j);

    let mut board = DynamicMap2d::new_with(0, input.n);
    let mut used = DynamicMap2d::new_with(false, input.n);
    let mut left = 1_i64;
    let mut right = input.n2 as i64;
    let mut Q1 = VecDeque::new();
    let mut Q2 = VecDeque::new();
    Q1.push_back(start);
    Q2.push_back(goal);
    let mut cnt = 0;

    while left <= right {
        let L = Q1.len();
        for _ in 0..L {
            let pos = Q1.pop_front().unwrap();
            if !used[pos] {
                Q1.push_back(pos);
            }
        }
        let L = Q2.len();
        for _ in 0..L {
            let pos = Q2.pop_front().unwrap();
            if !used[pos] {
                Q2.push_back(pos);
            }
        }

        if Q1.is_empty() {
            Q1 = Q2
                .clone()
                .into_iter()
                .collect_vec()
                .into_iter()
                .rev()
                .collect();
        }
        if Q2.is_empty() {
            Q2 = Q1
                .clone()
                .into_iter()
                .collect_vec()
                .into_iter()
                .rev()
                .collect();
        }

        if cnt % 2 == 0 {
            while let Some(pos) = Q1.pop_front() {
                if used[pos] {
                    continue;
                }
                board[pos] = left;
                used[pos] = true;
                for (_, nxt) in input.legal_actions[pos].iter() {
                    if !used[*nxt] {
                        Q1.push_back(*nxt);
                    }
                }
                break;
            }
            left += 1;
        } else {
            while let Some(pos) = Q2.pop_front() {
                if used[pos] {
                    continue;
                }
                board[pos] = right;
                used[pos] = true;
                for (_, nxt) in input.legal_actions[pos].iter() {
                    if !used[*nxt] {
                        Q2.push_back(*nxt);
                    }
                }
                break;
            }
            right -= 1;
        }
        cnt += 1;
    }
    #[cfg(feature = "local")]
    visualizer::vis(input.n, &input.vs, &input.hs, &board.to_2d_vec());
    (board, start, goal)
}

fn annealing(
    input: &Input,
    board: DynamicMap2d<i64>,
    rng: &mut rand_pcg::Pcg64Mcg,
    time_keeper: &TimeKeeper,
) -> DynamicMap2d<i64> {
    let mut state = AnnealingState::new(board, input);
    // eprintln!("Init cost: {}", state.cost);
    let mut best_state = state.clone();

    let T0 = (4 * input.n.pow(2)) as f64;
    let T1 = 1.0;
    let time_limit = 1.2;

    while time_keeper.get_time() < time_limit {
        let i1 = rng.gen_range(0..input.n);
        let j1 = rng.gen_range(0..input.n);
        let pos1 = Coord::new(i1, j1);
        let i2 = rng.gen_range(0..input.n);
        let j2 = rng.gen_range(0..input.n);
        let pos2 = Coord::new(i2, j2);
        if pos1 == pos2 {
            continue;
        }
        let diff = state.calc_diff_cost(pos1, pos2, input);
        let temp = T0 + (T1 - T0) * time_keeper.get_time() / time_limit;
        if diff <= 0 || rng.gen_bool((-diff as f64 / temp).exp()) {
            state.swap(pos1, pos2);
            state.cost += diff;
        }
        if state.cost < best_state.cost {
            best_state = state.clone();
            best_state.cost = state.cost;
        }
    }
    // let score = calc_score(input.cost, best_state.cost);
    // eprintln!("Ideal score: {}", score);
    #[cfg(feature = "local")]
    visualizer::vis(input.n, &input.vs, &input.hs, &best_state.board.to_2d_vec());
    best_state.board
}

fn calc_cost(board: &DynamicMap2d<i64>, input: &Input) -> i64 {
    let mut cost = 0;
    for i in 0..input.n {
        for j in 0..input.n {
            let coord = Coord::new(i, j);
            let coord_down = Coord::new(i + 1, j);
            let coord_right = Coord::new(i, j + 1);

            if i + 1 < input.n && input.hs[i][j] == '0' {
                cost += (board[coord] - board[coord_down]).pow(2);
            }
            if j + 1 < input.n && input.vs[i][j] == '0' {
                cost += (board[coord] - board[coord_right]).pow(2);
            }
        }
    }
    cost
}

fn calc_score(init_cost: i64, cost: i64) -> i64 {
    let score = (init_cost as f64).log2() - (cost as f64).log2();
    let mut score = (1e6 * score).round() as i64;
    score = score.max(1);
    score
}

#[derive(Debug, Clone, Default)]
struct State {
    n: usize,
    n2: usize,
    init_pos1: Coord,
    init_pos2: Coord,
    pos1: Coord,
    pos2: Coord,
    board: DynamicMap2d<i64>,
    actions: Vec<usize>,
}

impl State {
    fn new(input: &Input, target: &Target) -> State {
        State {
            n: input.n,
            n2: input.n2,
            init_pos1: target.start,
            init_pos2: target.goal,
            pos1: target.start,
            pos2: target.goal,
            board: input.board.clone(),
            actions: vec![0],
        }
    }
    fn swap(&mut self, pos1: Coord, pos2: Coord) {
        let tmp = self.board[pos1];
        self.board[pos1] = self.board[pos2];
        self.board[pos2] = tmp;
    }
    fn search_different(
        &self,
        left: usize,
        right: usize,
        target: &Target,
    ) -> (usize, DynamicMap2d<u8>) {
        let mut different = DynamicMap2d::new_with(0, self.n);
        let mid = (left + right) / 2;
        let mut cnt = 0;
        for i in 0..self.n {
            for j in 0..self.n {
                let coord = Coord::new(i, j);
                let target_val = target.board[coord] as usize - 1;
                let now_val = self.board[coord] as usize - 1;
                if left <= target_val && target_val < mid && mid <= now_val && now_val < right {
                    different[coord] = 1;
                    cnt += 1;
                }
                if left <= now_val && now_val < mid && mid <= target_val && target_val < right {
                    different[coord] = 2;
                    cnt += 1;
                }
            }
        }
        cnt /= 2;
        (cnt, different)
    }
    fn search_different_by_group(
        &self,
        left: usize,
        right: usize,
        target: &Target,
    ) -> (Vec<(usize, usize)>, DynamicMap2d<u8>) {
        let mut different = DynamicMap2d::new_with(0, self.n);
        let mid = (left + right) / 2;
        let mut counts = vec![(0, 0); target.group_num];
        for i in 0..self.n {
            for j in 0..self.n {
                let coord = Coord::new(i, j);
                let target_val = target.board[coord] as usize - 1;
                let now_val = self.board[coord] as usize - 1;
                let g = target.group_board[coord];
                if left <= target_val && target_val < mid && mid <= now_val && now_val < right {
                    different[coord] = 1;
                    counts[g].0 += 1;
                }
                if left <= now_val && now_val < mid && mid <= target_val && target_val < right {
                    different[coord] = 2;
                    counts[g].1 += 1;
                }
            }
        }
        (counts, different)
    }
    fn quick_sort(&mut self, input: &Input, target: &Target) {
        let mut Q = vec![];
        let mut next_Q = vec![];
        let mut cnt = 0;
        Q.push((0, self.n2));
        while let Some((left, right)) = Q.pop() {
            if left == right {
                continue;
            }
            self.tsp(input, left, right, target);
            if self.actions.len() > 3 * 4 * self.n2 {
                break;
            }
            if right - left > 1 {
                let mid = (left + right) / 2;
                next_Q.push((left, mid));
                next_Q.push((mid, right));
            }
            if Q.is_empty() {
                std::mem::swap(&mut Q, &mut next_Q);
                Q.sort();
                if cnt % 2 == 0 {
                    Q.reverse();
                }
                cnt += 1;
            }
        }
    }
    fn quick_sort_by_group(&mut self, input: &Input, target: &Target) {
        let mut Q = vec![];
        let mut next_Q = vec![];
        let mut cnt = 0;
        Q.push((0, self.n2));
        while let Some((left, right)) = Q.pop() {
            if left == right {
                continue;
            }
            self.tsp_by_group(input, left, right, target);
            if self.actions.len() > 3 * 4 * self.n2 {
                break;
            }
            if right - left > 1 {
                let mid = (left + right) / 2;
                next_Q.push((left, mid));
                next_Q.push((mid, right));
            }
            if Q.is_empty() {
                std::mem::swap(&mut Q, &mut next_Q);
                Q.sort();
                if cnt % 2 == 1 {
                    Q.reverse();
                }
                cnt += 1;
            }
        }
    }
    fn tsp(&mut self, input: &Input, left: usize, right: usize, target: &Target) {
        let (cnt, mut different) = self.search_different(left, right, target);
        for _ in 0..cnt {
            let mut actions1 = self.bfs(input, self.pos1, 1, &mut different);
            let mut actions2 = self.bfs(input, self.pos2, 2, &mut different);
            assert!(!(actions1.is_empty() && actions2.is_empty()));
            while actions1.len() < actions2.len() {
                actions1.push(DIRS_MAP[&'.']);
            }
            while actions1.len() > actions2.len() {
                actions2.push(DIRS_MAP[&'.']);
            }
            for i in 0..actions1.len() {
                self.actions.push(actions1[i]);
                self.pos1 = self.pos1 + DIJ_DIFF[actions1[i]];
                self.actions.push(actions2[i]);
                self.pos2 = self.pos2 + DIJ_DIFF[actions2[i]];
                self.actions.push(0);
            }
            *self.actions.last_mut().unwrap() = 1;
            self.swap(self.pos1, self.pos2);
        }
    }
    fn bitdp(&self, pos: Coord, go_group: Vec<usize>, target: &Target) -> Vec<usize> {
        let INF = 1_usize << 60;
        let MAX = 1 << target.group_num;
        let mut dp = vec![vec![INF; target.group_num]; MAX];
        let start = target.group_board[pos];
        dp[1 << start][start] = 0;
        for s in 1..MAX {
            for &frm in go_group.iter() {
                if s & (1 << frm) == 0 {
                    continue;
                }
                for &to in go_group.iter() {
                    if s & (1 << to) == 0 {
                        continue;
                    }
                    let bs = s ^ (1 << to);
                    dp[s][to] = dp[s][to].min(dp[bs][frm] + target.group_dist[frm][to]);
                }
            }
        }

        let mut d_min = INF;
        let mut frm = 0;
        let mut status = 0;
        for &g in go_group.iter() {
            status ^= 1 << g;
        }
        for (i, &d) in dp[status].iter().enumerate() {
            if d < d_min {
                d_min = d;
                frm = i;
            }
        }
        let mut route = vec![frm];
        while dp[status][frm] != 0 {
            for &to in go_group.iter() {
                let bs = status ^ (1 << frm);
                if dp[bs][to] + target.group_dist[frm][to] == dp[status][frm] {
                    route.push(to);
                    status = bs;
                    frm = to;
                    break;
                }
            }
        }
        route.reverse();
        route
    }
    fn tsp_by_group(&mut self, input: &Input, left: usize, right: usize, target: &Target) {
        let (mut counts, mut different) = self.search_different_by_group(left, right, target);
        let mut cnt = 0;
        for (c, _) in counts.iter() {
            cnt += c;
        }

        let mut go_group1 = vec![];
        let mut go_group2 = vec![];
        go_group1.push(target.group_board[self.pos1]);
        go_group2.push(target.group_board[self.pos2]);
        for g in 0..target.group_num {
            if counts[g].0 > 0 {
                go_group1.push(g);
            }
            if counts[g].1 > 0 {
                go_group2.push(g);
            }
        }
        go_group1.sort();
        go_group1.dedup();
        go_group2.sort();
        go_group2.dedup();

        let route1 = self.bitdp(self.pos1, go_group1, target);
        let route2 = self.bitdp(self.pos2, go_group2, target);
        let mut group_idx1 = 0;
        let mut group_idx2 = 0;
        for _ in 0..cnt {
            while counts[route1[group_idx1]].0 == 0 {
                group_idx1 += 1;
            }
            while counts[route2[group_idx2]].1 == 0 {
                group_idx2 += 1;
            }
            let g1 = route1[group_idx1];
            let g2 = route2[group_idx2];
            let mut actions1 = self.bfs_by_group(input, self.pos1, 1, &mut different, g1, target);
            counts[g1].0 -= 1;
            let mut actions2 = self.bfs_by_group(input, self.pos2, 2, &mut different, g2, target);
            counts[g2].1 -= 1;
            assert!(!(actions1.is_empty() && actions2.is_empty()));
            while actions1.len() < actions2.len() {
                actions1.push(DIRS_MAP[&'.']);
            }
            while actions1.len() > actions2.len() {
                actions2.push(DIRS_MAP[&'.']);
            }
            for i in 0..actions1.len() {
                self.actions.push(actions1[i]);
                self.pos1 = self.pos1 + DIJ_DIFF[actions1[i]];
                self.actions.push(actions2[i]);
                self.pos2 = self.pos2 + DIJ_DIFF[actions2[i]];
                self.actions.push(0);
            }
            *self.actions.last_mut().unwrap() = 1;
            self.swap(self.pos1, self.pos2);
        }
    }
    fn bfs(
        &self,
        input: &Input,
        st: Coord,
        search: u8,
        different: &mut DynamicMap2d<u8>,
    ) -> Vec<usize> {
        let mut dist: FxHashMap<Coord, usize> = FxHashMap::default();
        dist.insert(st, 0);
        let mut Q = VecDeque::new();
        Q.push_back((st, vec![]));
        while let Some((pos, actions)) = Q.pop_front() {
            for &(dir, nxt) in input.legal_actions[pos].iter() {
                if !dist.contains_key(&nxt) || dist[&pos] + 1 < dist[&nxt] {
                    let mut nxt_actions = actions.clone();
                    nxt_actions.push(dir);
                    if different[nxt] == search {
                        different[nxt] = 0;
                        return nxt_actions;
                    }
                    dist.insert(nxt, dist[&pos] + 1);
                    Q.push_back((nxt, nxt_actions));
                }
            }
        }
        vec![]
    }
    fn bfs_by_group(
        &self,
        input: &Input,
        st: Coord,
        search: u8,
        different: &mut DynamicMap2d<u8>,
        group: usize,
        target: &Target,
    ) -> Vec<usize> {
        let mut dist: FxHashMap<Coord, usize> = FxHashMap::default();
        dist.insert(st, 0);
        let mut Q = VecDeque::new();
        Q.push_back((st, vec![]));
        while let Some((pos, actions)) = Q.pop_front() {
            for &(dir, nxt) in input.legal_actions[pos].iter() {
                if !dist.contains_key(&nxt) || dist[&pos] + 1 < dist[&nxt] {
                    let mut nxt_actions = actions.clone();
                    nxt_actions.push(dir);
                    if different[nxt] == search && target.group_board[nxt] == group {
                        different[nxt] = 0;
                        return nxt_actions;
                    }
                    dist.insert(nxt, dist[&pos] + 1);
                    Q.push_back((nxt, nxt_actions));
                }
            }
        }
        vec![]
    }
    #[fastout]
    fn output(&mut self) {
        self.actions.push(DIRS_MAP[&'.']);
        self.actions.push(DIRS_MAP[&'.']);
        self.actions.truncate(3 * 4 * self.n2);
        println!(
            "{} {} {} {}",
            self.init_pos1.row, self.init_pos1.col, self.init_pos2.row, self.init_pos2.col
        );
        // ターン毎にスワップ、高橋君移動、青木移動のアクションを同時に保存できないので、
        // タプルは使用せずに、1次元配列に保存した。
        // mod3=0: スワップ
        // mod3=1: 高橋君移動
        // mod3=2: 青木君移動
        let mut i = 0;
        while i < self.actions.len() {
            let is_swap = self.actions[i];
            let dir1 = self.actions[i + 1];
            let dir2 = self.actions[i + 2];
            println!("{} {} {}", is_swap, DIRS[dir1], DIRS[dir2]);
            i += 3;
        }
    }
}

#[derive(Debug, Clone, Default)]
struct AnnealingState {
    board: DynamicMap2d<i64>,
    cost: i64,
}

impl AnnealingState {
    fn new(board: DynamicMap2d<i64>, input: &Input) -> AnnealingState {
        let cost = calc_cost(&board, input);
        AnnealingState { board, cost }
    }
    fn swap(&mut self, pos1: Coord, pos2: Coord) {
        let tmp = self.board[pos1];
        self.board[pos1] = self.board[pos2];
        self.board[pos2] = tmp;
    }
    fn calc_diff_cost(&self, pos1: Coord, pos2: Coord, input: &Input) -> i64 {
        // スワップする周辺だけ差分更新
        let mut before = 0;
        for nxt in input.legal_actions[pos1].iter() {
            before += (self.board[pos1] - self.board[nxt.1]).pow(2);
        }
        for nxt in input.legal_actions[pos2].iter() {
            before += (self.board[pos2] - self.board[nxt.1]).pow(2);
        }
        let mut after = 0;
        for nxt in input.legal_actions[pos1].iter() {
            if nxt.1 == pos2 {
                after += (self.board[pos2] - self.board[pos1]).pow(2);
            } else {
                after += (self.board[pos2] - self.board[nxt.1]).pow(2);
            }
        }
        for nxt in input.legal_actions[pos2].iter() {
            if nxt.1 == pos1 {
                after += (self.board[pos1] - self.board[pos2]).pow(2);
            } else {
                after += (self.board[pos1] - self.board[nxt.1]).pow(2);
            }
        }
        after - before
    }
}

const DIRS: [char; 5] = ['U', 'D', 'L', 'R', '.'];
const DIJ: [(usize, usize); 5] = [(!0, 0), (1, 0), (0, !0), (0, 1), (0, 0)];
const DIJ_DIFF: [CoordDiff; 5] = [
    CoordDiff::new(!0, 0),
    CoordDiff::new(1, 0),
    CoordDiff::new(0, !0),
    CoordDiff::new(0, 1),
    CoordDiff::new(0, 0),
];
const DIRS_REVERSE: [usize; 5] = [1, 0, 3, 2, 4];

lazy_static::lazy_static! {
    static ref DIRS_MAP: HashMap<char, usize> = {
        let mut mp = HashMap::new();
        for (i,dir) in DIRS.iter().enumerate() {
            mp.insert(*dir, i);
        }
        mp
    };
}

struct Input {
    t: usize,
    n: usize,
    n2: usize,
    cost: i64,
    board: DynamicMap2d<i64>,
    legal_actions: DynamicMap2d<Vec<(usize, Coord)>>,
    vs: Vec<Vec<char>>,
    hs: Vec<Vec<char>>,
}

fn read_input() -> Input {
    input! {
        t: usize,
        n: usize,
        vs: [Chars; n],
        hs: [Chars; n - 1],
        board2: [[i64; n]; n]
    }

    let mut board = DynamicMap2d::new_with(0, n);
    let mut legal_actions = DynamicMap2d::new_with(vec![], n);
    let mut cost = 0;

    for i in 0..n {
        for j in 0..n {
            let coord = Coord::new(i, j);
            let coord_down = Coord::new(i + 1, j);
            let coord_right = Coord::new(i, j + 1);

            board[coord] = board2[i][j];

            if i + 1 < n && hs[i][j] == '0' {
                legal_actions[coord].push((DIRS_MAP[&'D'], coord_down));
                legal_actions[coord_down].push((DIRS_MAP[&'U'], coord));
                cost += (board2[i][j] - board2[i + 1][j]).pow(2);
            }
            if j + 1 < n && vs[i][j] == '0' {
                legal_actions[coord].push((DIRS_MAP[&'R'], coord_right));
                legal_actions[coord_right].push((DIRS_MAP[&'L'], coord));
                cost += (board2[i][j] - board2[i][j + 1]).pow(2);
            }
        }
    }

    Input {
        t,
        n,
        n2: n * n,
        cost,
        board,
        legal_actions,
        vs,
        hs,
    }
}

#[derive(Debug, Clone)]
struct TimeKeeper {
    start_time: std::time::Instant,
    time_threshold: f64,
}

impl TimeKeeper {
    fn new(time_threshold: f64) -> Self {
        TimeKeeper {
            start_time: std::time::Instant::now(),
            time_threshold,
        }
    }
    #[inline]
    fn isTimeOver(&self) -> bool {
        let elapsed_time = self.start_time.elapsed().as_nanos() as f64 * 1e-9;
        #[cfg(feature = "local")]
        {
            elapsed_time * 0.55 >= self.time_threshold
        }
        #[cfg(not(feature = "local"))]
        {
            elapsed_time >= self.time_threshold
        }
    }
    #[inline]
    fn get_time(&self) -> f64 {
        let elapsed_time = self.start_time.elapsed().as_nanos() as f64 * 1e-9;
        #[cfg(feature = "local")]
        {
            elapsed_time * 0.55
        }
        #[cfg(not(feature = "local"))]
        {
            elapsed_time
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Coord {
    row: usize,
    col: usize,
}

impl Coord {
    pub fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }
    pub fn in_map(&self, height: usize, width: usize) -> bool {
        self.row < height && self.col < width
    }
    pub fn to_index(&self, width: usize) -> CoordIndex {
        CoordIndex(self.row * width + self.col)
    }
}

impl std::ops::Add<CoordDiff> for Coord {
    type Output = Coord;
    fn add(self, rhs: CoordDiff) -> Self::Output {
        Coord::new(
            self.row.wrapping_add_signed(rhs.dr),
            self.col.wrapping_add_signed(rhs.dc),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoordDiff {
    dr: isize,
    dc: isize,
}

impl CoordDiff {
    pub const fn new(dr: isize, dc: isize) -> Self {
        Self { dr, dc }
    }
}

pub const ADJ: [CoordDiff; 4] = [
    CoordDiff::new(1, 0),
    CoordDiff::new(!0, 0),
    CoordDiff::new(0, 1),
    CoordDiff::new(0, !0),
];

pub struct CoordIndex(pub usize);

impl CoordIndex {
    pub fn new(index: usize) -> Self {
        Self(index)
    }
    pub fn to_coord(&self, width: usize) -> Coord {
        Coord {
            row: self.0 / width,
            col: self.0 % width,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DynamicMap2d<T> {
    pub size: usize,
    map: Vec<T>,
}

impl<T> DynamicMap2d<T> {
    pub fn new(map: Vec<T>, size: usize) -> Self {
        assert_eq!(size * size, map.len());
        Self { size, map }
    }
}

impl<T: Clone> DynamicMap2d<T> {
    pub fn new_with(v: T, size: usize) -> Self {
        let map = vec![v; size * size];
        Self::new(map, size)
    }
    pub fn to_2d_vec(&self) -> Vec<Vec<T>> {
        let mut ret = vec![vec![]; self.size];
        for i in 0..self.map.len() {
            let row = i / self.size;
            ret[row].push(self.map[i].clone());
        }
        ret
    }
}

impl<T> std::ops::Index<Coord> for DynamicMap2d<T> {
    type Output = T;

    #[inline]
    fn index(&self, coordinate: Coord) -> &Self::Output {
        &self[coordinate.to_index(self.size)]
    }
}

impl<T> std::ops::IndexMut<Coord> for DynamicMap2d<T> {
    #[inline]
    fn index_mut(&mut self, coordinate: Coord) -> &mut Self::Output {
        let size = self.size;
        &mut self[coordinate.to_index(size)]
    }
}

impl<T> std::ops::Index<CoordIndex> for DynamicMap2d<T> {
    type Output = T;

    fn index(&self, index: CoordIndex) -> &Self::Output {
        unsafe { self.map.get_unchecked(index.0) }
    }
}

impl<T> std::ops::IndexMut<CoordIndex> for DynamicMap2d<T> {
    #[inline]
    fn index_mut(&mut self, index: CoordIndex) -> &mut Self::Output {
        unsafe { self.map.get_unchecked_mut(index.0) }
    }
}

#[cfg(feature = "local")]
mod visualizer {
    use svg::node::element::path::Data;
    use svg::node::element::{Circle, Group, Line, Path, Rectangle, Style, Text, Title};
    use svg::node::Text as TextContent;
    use svg::Document;

    const MARGIN: f32 = 10.0;

    pub fn doc(height: f32, width: f32) -> Document {
        Document::new()
            .set(
                "viewBox",
                (
                    -MARGIN,
                    -MARGIN,
                    width + 2.0 * MARGIN,
                    height + 2.0 * MARGIN,
                ),
            )
            .set("width", width + MARGIN)
            .set("height", height + MARGIN)
            .set("style", "background-color:#F2F3F5")
    }

    pub fn rect(x: f32, y: f32, w: f32, h: f32, fill: &str) -> Rectangle {
        Rectangle::new()
            .set("x", x)
            .set("y", y)
            .set("width", w)
            .set("height", h)
            .set("fill", fill)
    }

    pub fn cir(x: usize, y: usize, r: usize, fill: &str) -> Circle {
        Circle::new()
            .set("cx", x)
            .set("cy", y)
            .set("r", r)
            .set("fill", fill)
    }

    pub fn lin(x1: usize, y1: usize, x2: usize, y2: usize, color: &str) -> Line {
        Line::new()
            .set("x1", x1)
            .set("y1", y1)
            .set("x2", x2)
            .set("y2", y2)
            .set("stroke", color)
            .set("stroke-width", 3)
            .set("stroke-linecap", "round")
            .set("stroke-linecap", "round")
            .set("marker-end", "url(#arrowhead)")
        // .set("stroke-dasharray", 5)
    }

    pub fn arrow(doc: Document) -> Document {
        doc.add(TextContent::new(
        r#"<defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="3" refY="2" orient="auto">
                <polygon points="0 0, 4 2, 0 4" fill="lightgray"/>
            </marker>
        </defs>"#,
    ))
    }

    pub fn txt(x: f32, y: f32, text: &str) -> Text {
        Text::new()
            .add(TextContent::new(text))
            .set("x", x)
            .set("y", y)
            .set("fill", "black")
    }

    // 0 <= val <= 1
    pub fn color(mut val: f64) -> String {
        val = val.min(1.0);
        val = val.max(0.0);
        let (r, g, b) = if val < 0.5 {
            let x = val * 2.0;
            (
                30. * (1.0 - x) + 144. * x,
                144. * (1.0 - x) + 255. * x,
                255. * (1.0 - x) + 30. * x,
            )
        } else {
            let x = val * 2.0 - 1.0;
            (
                144. * (1.0 - x) + 255. * x,
                255. * (1.0 - x) + 30. * x,
                30. * (1.0 - x) + 70. * x,
            )
        };
        format!(
            "#{:02x}{:02x}{:02x}",
            r.round() as i32,
            g.round() as i32,
            b.round() as i32
        )
    }

    pub fn group(title: String) -> Group {
        Group::new().add(Title::new().add(TextContent::new(title)))
    }

    pub fn partition(mut doc: Document, h: &[Vec<char>], v: &[Vec<char>], size: f32) -> Document {
        let H = v.len();
        let W = h[0].len();
        for i in 0..H + 1 {
            for j in 0..W {
                // Entrance
                // if i == 0 && j == ENTRANCE {
                //     continue;
                // }
                if (i == 0 || i == H) || h[i - 1][j] == '1' {
                    let data = Data::new()
                        .move_to((size * j as f32, size * i as f32))
                        .line_by((size * 1.0, 0));
                    let p = Path::new()
                        .set("d", data)
                        .set("stroke", "black")
                        .set("stroke-width", 3.0)
                        .set("stroke-linecap", "round");
                    doc = doc.add(p);
                }
            }
        }
        for j in 0..W + 1 {
            for i in 0..H {
                // Entrance
                // if j == 0 && i == ENTRANCE {
                //     continue;
                // }
                if (j == 0 || j == W) || v[i][j - 1] == '1' {
                    let data = Data::new()
                        .move_to((size * j as f32, size * i as f32))
                        .line_by((0, size * 1.0));
                    let p = Path::new()
                        .set("d", data)
                        .set("stroke", "black")
                        .set("stroke-width", 3.0)
                        .set("stroke-linecap", "round");
                    doc = doc.add(p);
                }
            }
        }
        doc
    }

    pub fn vis(N: usize, vs: &[Vec<char>], hs: &[Vec<char>], board: &[Vec<i64>]) {
        let height = 800.0;
        let width = 800.0;
        let N2 = N * N;
        let d = height / N as f32;
        let mut doc = doc(height, width);
        doc = doc.add(Style::new(format!(
            "text {{text-anchor: middle; dominant-baseline: central; font-size: {}}}",
            10
        )));

        for i in 0..N {
            for j in 0..N {
                let rec = rect(
                    j as f32 * d,
                    i as f32 * d,
                    d,
                    d,
                    &color(board[i][j] as f64 / N2 as f64),
                );
                let text = txt(
                    j as f32 * d + d / 2.0,
                    i as f32 * d + d / 2.0,
                    &board[i][j].to_string(),
                );
                let mut grp = group(format!("(i, j) = ({}, {})\n{}", i, j, board[i][j]));
                grp = grp.add(rec);
                if N <= 20 {
                    grp = grp.add(text);
                }
                doc = doc.add(grp);
            }
        }
        doc = partition(doc, hs, vs, d);
        let vis = format!("<html><body>{}</body></html>", doc);
        std::fs::write("vis.html", vis).unwrap();
    }

    pub fn vis_grp(
        N: usize,
        vs: &[Vec<char>],
        hs: &[Vec<char>],
        board: &[Vec<usize>],
        group_num: usize,
    ) {
        let height = 800.0;
        let width = 800.0;
        let d = height / N as f32;
        let mut doc = doc(height, width);
        doc = doc.add(Style::new(format!(
            "text {{text-anchor: middle; dominant-baseline: central; font-size: {}}}",
            10
        )));

        for i in 0..N {
            for j in 0..N {
                let rec = rect(
                    j as f32 * d,
                    i as f32 * d,
                    d,
                    d,
                    &color(board[i][j] as f64 / group_num as f64),
                );
                let text = txt(
                    j as f32 * d + d / 2.0,
                    i as f32 * d + d / 2.0,
                    &board[i][j].to_string(),
                );
                let mut grp = group(format!("(i, j) = ({}, {})\n{}", i, j, board[i][j]));
                grp = grp.add(rec);
                if N <= 20 {
                    grp = grp.add(text);
                }
                doc = doc.add(grp);
            }
        }
        doc = partition(doc, hs, vs, d);
        let vis = format!("<html><body>{}</body></html>", doc);
        std::fs::write("group.html", vis).unwrap();
    }
}
