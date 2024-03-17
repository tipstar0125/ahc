#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::neg_multiply)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};

use proconio::{fastout, input, marker::Chars};
use rand::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Default)]
struct Solver {}
impl Solver {
    fn solve(&mut self) {
        let start = std::time::Instant::now();

        let time_keeper = TimeKeeper::new(1.98);
        let mut input = read_input();
        let mut rng = rand_pcg::Pcg64Mcg::new(12345);

        let mut init_state = State::default();
        let mut cnt = 0;

        while time_keeper.get_time() <= 0.1 {
            cnt += 1;
            let init_pos1 = init_pos(input.n, &mut rng, &input);
            let init_pos2 = init_pos(input.n, &mut rng, &input);
            let mut state = State::new(&mut input, init_pos1, init_pos2, &mut rng);
            state.init_solve(&input);
            if state.cost < init_state.cost {
                init_state = state;
            }
        }
        let mut best_state = init_state.clone();
        let mut state = init_state;

        while !time_keeper.isTimeOver() {}

        best_state.output();
        eprintln!("Cost: {}", best_state.cost);
        eprintln!("Score: {}", best_state.calc_score(input.cost));
        eprintln!("Count: {}", cnt);

        #[allow(unused_mut, unused_assignments)]
        let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            elapsed_time *= 0.55;
        }
        eprintln!("Elapsed: {}", (elapsed_time * 1000.0) as usize);
    }
}

fn init_pos(n: usize, rng: &mut rand_pcg::Pcg64Mcg, input: &Input) -> Coord {
    loop {
        let i = rng.gen_range(0..n);
        let j = rng.gen_range(0..n);
        let coord = Coord::new(i, j);
        if input.legal_actions[coord].len() > 1 {
            return coord;
        }
    }
}

#[derive(Debug, Clone)]
struct State {
    route1: Vec<Coord>,
    route2: Vec<Coord>,
    dir1: Vec<usize>,
    dir2: Vec<usize>,
    swaps: Vec<bool>,
    max_turn: usize,
    board: DynamicMap2d<isize>,
    cost: isize,
}

impl State {
    fn default() -> Self {
        State {
            route1: vec![],
            route2: vec![],
            dir1: vec![],
            dir2: vec![],
            swaps: vec![],
            max_turn: 0,
            board: DynamicMap2d::new_with(0, 1),
            cost: std::isize::MAX,
        }
    }
    fn new(
        input: &mut Input,
        init_pos1: Coord,
        init_pos2: Coord,
        rng: &mut rand_pcg::Pcg64Mcg,
    ) -> Self {
        fn dfs(
            pos: Coord,
            visited: &mut DynamicMap2d<bool>,
            route: &mut Vec<Coord>,
            dir: &mut Vec<usize>,
            input: &Input,
        ) {
            for (d, nxt) in &input.legal_actions[pos] {
                if visited[*nxt] {
                    continue;
                }
                route.push(*nxt);
                dir.push(*d);
                visited[*nxt] = true;
                dfs(*nxt, visited, route, dir, input);
                route.push(pos);
                dir.push(DIRS_REVERSE[*d]);
            }
        }

        let mut route1 = vec![init_pos1];
        let mut route2 = vec![init_pos2];
        let mut dir1 = vec![];
        let mut dir2 = vec![];
        let max_turn = 4 * input.n * input.n;

        while dir1.len() < max_turn {
            let mut visited = DynamicMap2d::new_with(false, input.n);
            visited[init_pos1] = true;
            dfs(init_pos1, &mut visited, &mut route1, &mut dir1, input);
            input.legal_actions[init_pos1].shuffle(rng);
        }

        dir1.truncate(max_turn);
        route1.truncate(max_turn);

        while dir2.len() < max_turn {
            let mut visited = DynamicMap2d::new_with(false, input.n);
            visited[init_pos2] = true;
            dfs(init_pos2, &mut visited, &mut route2, &mut dir2, input);
            input.legal_actions[init_pos2].shuffle(rng);
        }

        dir2.truncate(max_turn);
        route2.truncate(max_turn);

        State {
            route1,
            route2,
            dir1,
            dir2,
            swaps: vec![false; max_turn],
            max_turn,
            board: input.board.clone(),
            cost: input.cost,
        }
    }
    fn swap(&mut self, pos1: Coord, pos2: Coord) {
        let tmp = self.board[pos1];
        self.board[pos1] = self.board[pos2];
        self.board[pos2] = tmp;
    }
    fn swap_with_calc_cost_diff(
        &mut self,
        pos1: Coord,
        pos2: Coord,
        legal_actions: &DynamicMap2d<Vec<(usize, Coord)>>,
    ) -> isize {
        // スワップする周辺だけ差分更新
        let mut before = 0;
        for nxt in legal_actions[pos1].iter() {
            before += (self.board[pos1] - self.board[nxt.1]).pow(2);
        }
        for nxt in legal_actions[pos2].iter() {
            before += (self.board[pos2] - self.board[nxt.1]).pow(2);
        }
        self.swap(pos1, pos2);
        let mut after = 0;
        for nxt in legal_actions[pos1].iter() {
            after += (self.board[pos1] - self.board[nxt.1]).pow(2);
        }
        for nxt in legal_actions[pos2].iter() {
            after += (self.board[pos2] - self.board[nxt.1]).pow(2);
        }
        self.swap(pos1, pos2);
        after - before
    }
    fn init_solve(&mut self, input: &Input) {
        for i in 0..self.max_turn {
            let pos1 = self.route1[i];
            let pos2 = self.route2[i];
            let diff = self.swap_with_calc_cost_diff(pos1, pos2, &input.legal_actions);
            if diff < 0 {
                self.swap(pos1, pos2);
                self.cost += diff;
                self.swaps[i] = true;
            }
        }
    }
    #[fastout]
    fn output(&self) {
        let init_pos1 = self.route1[0];
        let init_pos2 = self.route2[0];
        println!(
            "{} {} {} {}",
            init_pos1.row, init_pos1.col, init_pos2.row, init_pos2.col
        );
        for i in 0..self.max_turn {
            println!(
                "{} {} {}",
                if self.swaps[i] { 1 } else { 0 },
                DIRS[self.dir1[i]],
                DIRS[self.dir2[i]]
            );
        }
    }
    fn calc_score(&self, init_cost: isize) -> usize {
        let score = (init_cost as f64).log2() - (self.cost as f64).log2();
        let mut score = (1e6 * score).round() as usize;
        score = score.max(1);
        score
    }
}

const DIRS: [char; 5] = ['U', 'D', 'L', 'R', '.'];
const DIJ: [(usize, usize); 5] = [(!0, 0), (1, 0), (0, !0), (0, 1), (0, 0)];
const DIRS_REVERSE: [usize; 4] = [1, 0, 3, 2];

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
    cost: isize,
    board: DynamicMap2d<isize>,
    legal_actions: DynamicMap2d<Vec<(usize, Coord)>>,
}

fn read_input() -> Input {
    input! {
        t: usize,
        n: usize,
        v: [Chars; n],
        h: [Chars; n - 1],
        board2: [[isize; n]; n]
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

            if i + 1 < n && h[i][j] == '0' {
                legal_actions[coord].push((DIRS_MAP[&'D'], coord_down));
                legal_actions[coord_down].push((DIRS_MAP[&'U'], coord));
                cost += (board2[i][j] - board2[i + 1][j]).pow(2);
            }
            if j + 1 < n && v[i][j] == '0' {
                legal_actions[coord].push((DIRS_MAP[&'R'], coord_right));
                legal_actions[coord_right].push((DIRS_MAP[&'L'], coord));
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

fn main() {
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(|| Solver::default().solve())
        .unwrap()
        .join()
        .unwrap();
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
