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
    let time_keeper = TimeKeeper::new(1.95);
    let input = read_input();
    let mut rng = rand_pcg::Pcg64Mcg::new(0);

    let mut best_state = State::default();

    while !time_keeper.isTimeOver() {
        let init_pos1 = init_pos(input.n, &mut rng);
        let init_pos2 = init_pos(input.n, &mut rng);
        let mut state = State::new(
            input.n,
            input.cost,
            init_pos1,
            init_pos2,
            input.board.clone(),
        );

        while !state.is_done() {
            let diff = state.swap_with_calc_cost_diff(&input.legal_actions);
            let mut is_swap = 1;
            if diff < 0 {
                state.cost += diff;
            } else {
                state.swap();
                is_swap = 0;
            }
            let idx1 = rng.gen_range(0..input.legal_actions[state.pos1].len());
            let idx2 = rng.gen_range(0..input.legal_actions[state.pos2].len());
            let (dir1, next1) = input.legal_actions[state.pos1][idx1];
            let (dir2, next2) = input.legal_actions[state.pos2][idx2];
            state.actions.push((is_swap, dir1, dir2));
            state.pos1 = next1;
            state.pos2 = next2;
        }
        if state.cost < best_state.cost {
            best_state = state;
        }
    }

    best_state.output();
    eprintln!("Score: {}", best_state.calc_score(input.cost));
}

fn init_pos(n: usize, rng: &mut rand_pcg::Pcg64Mcg) -> Coord {
    let i = rng.gen_range(0..n);
    let j = rng.gen_range(0..n);
    Coord { row: i, col: j }
}

const DIRS: [char; 5] = ['U', 'D', 'L', 'R', '.'];
const DIJ: [(usize, usize); 5] = [(!0, 0), (1, 0), (0, !0), (0, 1), (0, 0)];

struct State {
    n: usize,
    init_pos1: Coord,
    init_pos2: Coord,
    pos1: Coord,
    pos2: Coord,
    cost: i32,
    board: DynamicMap2d<i32>,
    actions: Vec<(usize, usize, usize)>,
}

impl State {
    fn default() -> Self {
        State {
            n: 1,
            init_pos1: Coord { row: 0, col: 0 },
            init_pos2: Coord { row: 0, col: 0 },
            pos1: Coord { row: 0, col: 0 },
            pos2: Coord { row: 0, col: 0 },
            cost: std::i32::MAX,
            board: DynamicMap2d::new_with(0, 1),
            actions: vec![],
        }
    }
    fn new(n: usize, cost: i32, pos1: Coord, pos2: Coord, board: DynamicMap2d<i32>) -> Self {
        State {
            n,
            init_pos1: pos1,
            init_pos2: pos2,
            pos1,
            pos2,
            cost,
            board,
            actions: vec![],
        }
    }
    fn swap(&mut self) {
        let tmp = self.board[self.pos1];
        self.board[self.pos1] = self.board[self.pos2];
        self.board[self.pos2] = tmp;
    }
    fn swap_with_calc_cost_diff(
        &mut self,
        legal_actions: &DynamicMap2d<Vec<(usize, Coord)>>,
    ) -> i32 {
        let mut before = 0;
        for nxt in legal_actions[self.pos1].iter() {
            before += (self.board[self.pos1] - self.board[nxt.1]).pow(2);
        }
        for nxt in legal_actions[self.pos2].iter() {
            before += (self.board[self.pos2] - self.board[nxt.1]).pow(2);
        }
        self.swap();
        let mut after = 0;
        for nxt in legal_actions[self.pos1].iter() {
            after += (self.board[self.pos1] - self.board[nxt.1]).pow(2);
        }
        for nxt in legal_actions[self.pos2].iter() {
            after += (self.board[self.pos2] - self.board[nxt.1]).pow(2);
        }
        after - before
    }
    #[fastout]
    fn output(&self) {
        println!(
            "{} {} {} {}",
            self.init_pos1.row, self.init_pos1.col, self.init_pos2.row, self.init_pos2.col
        );
        for (is_swap, dir1, dir2) in self.actions.iter() {
            println!("{} {} {}", is_swap, DIRS[*dir1], DIRS[*dir2]);
        }
    }
    fn calc_score(&self, init_cost: i32) -> usize {
        let score = (init_cost as f64).log2() - (self.cost as f64).log2();
        let mut score = (1e6 * score).round() as usize;
        score = score.max(1);
        score
    }
    fn is_done(&self) -> bool {
        self.actions.len() >= 4 * self.n * self.n
    }
}

struct Input {
    t: usize,
    n: usize,
    cost: i32,
    board: DynamicMap2d<i32>,
    legal_actions: DynamicMap2d<Vec<(usize, Coord)>>,
}

fn read_input() -> Input {
    input! {
        t: usize,
        n: usize,
        v: [Chars; n],
        h: [Chars; n - 1],
        board2: [[i32; n]; n]
    }

    let mut dirs_map = HashMap::new();
    for (i, &dir) in DIRS.iter().enumerate() {
        dirs_map.insert(dir, i);
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
            legal_actions[coord].push((dirs_map[&'.'], coord));

            if i + 1 < n && h[i][j] == '0' {
                legal_actions[coord].push((dirs_map[&'D'], coord_down));
                legal_actions[coord_down].push((dirs_map[&'U'], coord));
                cost += (board2[i][j] - board2[i + 1][j]).pow(2);
            }
            if j + 1 < n && v[i][j] == '0' {
                legal_actions[coord].push((dirs_map[&'R'], coord_right));
                legal_actions[coord_right].push((dirs_map[&'L'], coord));
                cost += (board2[i][j] - board2[i][j + 1]).pow(2);
            }
        }
    }

    Input {
        t,
        n,
        cost,
        board,
        legal_actions,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone)]
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
