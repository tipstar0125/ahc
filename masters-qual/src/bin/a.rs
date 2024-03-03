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
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque},
};

use itertools::Itertools;
use proconio::{
    fastout, input,
    marker::{Chars, Usize1},
};
use rand::prelude::*;
use superslice::Ext;

fn main() {
    let start = std::time::Instant::now();

    solve();

    #[allow(unused_mut, unused_assignments)]
    let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
    #[cfg(feature = "local")]
    {
        eprintln!("Local Mode");
        elapsed_time *= 0.55;
    }
    eprintln!("Elapsed: {}", (elapsed_time * 1000.0) as usize);
}

fn solve() {
    let state = State::read();
    let N = state.N;
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);

    let time_keeper = TimeKeeper::new(1.95);
    #[allow(clippy::redundant_clone)]
    let mut best_state = state.clone();
    let mut best_score = 0;

    'outer: while !time_keeper.isTimeOver() {
        let mut now_state = state.clone();
        let a_pos = Coord {
            r: rng.gen_range(0..N),
            c: rng.gen_range(0..N),
        };
        let b_pos = Coord {
            r: rng.gen_range(0..N),
            c: rng.gen_range(0..N),
        };
        now_state.initial_pos(a_pos, b_pos);

        for _ in 0..4 * N * N {
            if time_keeper.isTimeOver() {
                break 'outer;
            }
            let diff_calc = now_state.calc_diff_score();
            let is_swap = diff_calc < 0;
            let legal_actions_a = now_state.legal_action_a();
            let legal_actions_b = now_state.legal_action_b();
            let dir_a = legal_actions_a[rng.gen_range(0..legal_actions_a.len())];
            let dir_b = legal_actions_b[rng.gen_range(0..legal_actions_b.len())];
            now_state.action_(is_swap, dir_a, dir_b);
        }
        let score = now_state.calc_score();
        if score > best_score {
            best_score = score;
            best_state = now_state;
        }
    }
    best_state.output();
    let score = best_state.calc_score();
    eprintln!("Score: {}", score);
}

const DIRS: [char; 5] = ['U', 'D', 'L', 'R', '.'];
const DIJ: [(usize, usize); 5] = [(!0, 0), (1, 0), (0, !0), (0, 1), (0, 0)];

#[derive(Debug, Clone, Copy)]
struct Coord {
    r: usize,
    c: usize,
}

#[derive(Debug, Clone)]
struct State {
    N: usize,
    t: usize,
    G: Vec<Vec<Vec<Coord>>>,
    board: Vec<Vec<isize>>,
    initial_a_pos: Coord,
    initial_b_pos: Coord,
    a_pos: Coord,
    b_pos: Coord,
    initial_D: isize,
    actions: Vec<(usize, usize, usize)>,
}

impl State {
    fn read() -> Self {
        input! {
            t: usize,
            N: usize,
            v: [Chars; N],
            h: [Chars; N - 1],
            board: [[isize; N]; N]
        }
        let mut G = vec![vec![vec![]; N]; N];

        for i in 0..N {
            for j in 0..N {
                if i + 1 < N && h[i][j] == '0' {
                    G[i][j].push(Coord { r: i + 1, c: j });
                    G[i + 1][j].push(Coord { r: i, c: j });
                }
                if j + 1 < N && v[i][j] == '0' {
                    G[i][j].push(Coord { r: i, c: j + 1 });
                    G[i][j + 1].push(Coord { r: i, c: j });
                }
            }
        }
        let initial_a_pos = Coord { r: 0, c: 0 };
        let initial_b_pos = Coord { r: 0, c: 0 };
        let a_pos = Coord { r: 0, c: 0 };
        let b_pos = Coord { r: 0, c: 0 };

        let mut initial_D = 0;
        for i in 0..N {
            for j in 0..N {
                for nxt in &G[i][j] {
                    let diff = board[i][j] - board[nxt.r][nxt.c];
                    initial_D += diff.pow(2);
                }
            }
        }
        initial_D /= 2;

        State {
            N,
            t,
            G,
            board,
            initial_a_pos,
            initial_b_pos,
            a_pos,
            b_pos,
            initial_D,
            actions: vec![],
        }
    }
    fn initial_pos(&mut self, a_pos: Coord, b_pos: Coord) {
        self.initial_a_pos = a_pos;
        self.initial_b_pos = b_pos;
        self.a_pos = a_pos;
        self.b_pos = b_pos;
    }
    fn action_(&mut self, is_swap: bool, dir_a: usize, dir_b: usize) {
        if is_swap {
            self.swap_();
        }
        self.move_to_a(dir_a);
        self.move_to_b(dir_b);

        let is_swap = if is_swap { 1 } else { 0 };
        self.actions.push((is_swap, dir_a, dir_b));
    }
    fn swap_(&mut self) {
        let tmp = self.board[self.a_pos.r][self.a_pos.c];
        self.board[self.a_pos.r][self.a_pos.c] = self.board[self.b_pos.r][self.b_pos.c];
        self.board[self.b_pos.r][self.b_pos.c] = tmp;
    }
    fn legal_action_a(&self) -> Vec<usize> {
        let mut actions = vec![];
        for nxt in &self.G[self.a_pos.r][self.a_pos.c] {
            let dir = decode_dir(&self.a_pos, nxt);
            actions.push(dir);
        }
        actions.push(4);
        actions
    }
    fn legal_action_b(&self) -> Vec<usize> {
        let mut actions = vec![];
        for nxt in &self.G[self.b_pos.r][self.b_pos.c] {
            let dir = decode_dir(&self.b_pos, nxt);
            actions.push(dir);
        }
        actions.push(4);
        actions
    }
    fn move_to_a(&mut self, dir: usize) {
        self.a_pos.r = self.a_pos.r.wrapping_add(DIJ[dir].0);
        self.a_pos.c = self.a_pos.c.wrapping_add(DIJ[dir].1);
    }
    fn move_to_b(&mut self, dir: usize) {
        self.b_pos.r = self.b_pos.r.wrapping_add(DIJ[dir].0);
        self.b_pos.c = self.b_pos.c.wrapping_add(DIJ[dir].1);
    }
    fn calc_diff_score(&mut self) -> isize {
        let mut before: HashMap<((usize, usize), (usize, usize)), isize> = HashMap::new();
        let r0 = self.a_pos.r;
        let c0 = self.a_pos.c;
        for &nxt in &self.G[r0][c0] {
            let r1 = nxt.r;
            let c1 = nxt.c;
            let diff = self.board[r0][c0] - self.board[r1][c1];
            *before.entry(((r0, c0), (r1, c1))).or_default() = diff.pow(2);
            *before.entry(((r1, c1), (r0, c0))).or_default() = diff.pow(2);
        }
        let r0 = self.b_pos.r;
        let c0 = self.b_pos.c;
        for &nxt in &self.G[r0][c0] {
            let r1 = nxt.r;
            let c1 = nxt.c;
            let diff = self.board[r0][c0] - self.board[r1][c1];
            *before.entry(((r0, c0), (r1, c1))).or_default() = diff.pow(2);
            *before.entry(((r1, c1), (r0, c0))).or_default() = diff.pow(2);
        }
        let mut before_D = 0;
        for (_, v) in before.iter() {
            before_D += v;
        }
        self.swap_();
        let mut after: HashMap<((usize, usize), (usize, usize)), isize> = HashMap::new();
        let r0 = self.a_pos.r;
        let c0 = self.a_pos.c;
        for &nxt in &self.G[r0][c0] {
            let r1 = nxt.r;
            let c1 = nxt.c;
            let diff = self.board[r0][c0] - self.board[r1][c1];
            *after.entry(((r0, c0), (r1, c1))).or_default() = diff.pow(2);
            *after.entry(((r1, c1), (r0, c0))).or_default() = diff.pow(2);
        }
        let r0 = self.b_pos.r;
        let c0 = self.b_pos.c;
        for &nxt in &self.G[r0][c0] {
            let r1 = nxt.r;
            let c1 = nxt.c;
            let diff = self.board[r0][c0] - self.board[r1][c1];
            *after.entry(((r0, c0), (r1, c1))).or_default() = diff.pow(2);
            *after.entry(((r1, c1), (r0, c0))).or_default() = diff.pow(2);
        }
        let mut after_D = 0;
        for (_, v) in after.iter() {
            after_D += v;
        }
        self.swap_();
        after_D - before_D
    }
    fn calc_score(&self) -> usize {
        let N = self.N;
        let mut D = 0;
        for i in 0..N {
            for j in 0..N {
                for nxt in &self.G[i][j] {
                    let diff = self.board[i][j] - self.board[nxt.r][nxt.c];
                    D += diff.pow(2);
                }
            }
        }
        D /= 2;
        let score = (self.initial_D as f64).log2() - (D as f64).log2();
        let mut score = (1e6 * score).round() as usize;
        score = score.max(1);
        score
    }
    #[fastout]
    fn output(&self) {
        println!(
            "{} {} {} {}",
            self.initial_a_pos.r, self.initial_a_pos.c, self.initial_b_pos.r, self.initial_b_pos.c
        );
        for (is_swap, dir_a, dir_b) in &self.actions {
            println!("{} {} {}", is_swap, DIRS[*dir_a], DIRS[*dir_b]);
        }
    }
}

fn decode_dir(pos0: &Coord, pos1: &Coord) -> usize {
    if pos0.r == pos1.r {
        if pos0.c < pos1.c {
            3
        } else {
            2
        }
    } else if pos0.r < pos1.r {
        1
    } else {
        0
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
