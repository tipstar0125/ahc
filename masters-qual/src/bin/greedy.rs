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

use amplify::confinement::Collection;
use proconio::{fastout, input, marker::Chars};
use rand::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

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

// 1ターン毎のアクションはスワップしてから移動をするが、
// 貪欲をする場合、移動をしてからスワップして改善するかどうかを判定するので、
// 最初のスワップはしない、最後の移動はしない、とした。

fn solve() {
    let time_keeper = TimeKeeper::new(1.98);
    let input = read_input();
    let mut rng = rand_pcg::Pcg64Mcg::new(12345);

    let mut best_state = State::default();
    let max_action = 4 * input.n * input.n * 3;
    let mut cnt = 0;
    let bfs_dist = if input.t == 19 { 7 } else { 10 };

    while !time_keeper.isTimeOver() {
        cnt += 1;
        // 初期位置はランダム生成
        let init_pos1 = init_pos(input.n, &mut rng);
        let init_pos2 = init_pos(input.n, &mut rng);
        let mut state = State::new(
            input.n,
            input.cost,
            init_pos1,
            init_pos2,
            input.board.clone(),
        );

        // 最初のスワップは行わない
        state.actions.push(0);

        while !state.is_done() {
            if time_keeper.isTimeOver() {
                break;
            }

            // BFSして、到達できる座標と距離、行き方を列挙
            let pos1_bfs_list = bfs(state.pos1, &input, bfs_dist);
            let pos2_bfs_list = bfs(state.pos2, &input, bfs_dist);
            let mut best_score = std::f64::MAX;
            let mut best_diff = 0;
            let mut best_actions1 = vec![];
            let mut best_actions2 = vec![];
            let mut best_nxt1 = Coord::new(0, 0);
            let mut best_nxt2 = Coord::new(0, 0);

            // BFSのすべての組み合わせに対して、Δcost/dist(負)が最小になる組合せを取得
            for (dist1, nxt1, actions1) in pos1_bfs_list.iter() {
                for (dist2, nxt2, actions2) in pos2_bfs_list.iter() {
                    if *dist1 == 0 && *dist2 == 1 {
                        continue;
                    }
                    let diff = state.swap_with_calc_cost_diff(*nxt1, *nxt2, &input.legal_actions);
                    // diff=0まで許容
                    if diff > 0 {
                        continue;
                    }
                    let mx = *dist1.max(dist2);
                    let score = (diff as f64) / (mx as f64);

                    // 改善する組合せがなくて、移動せずに終わるのはロスなので、スコアが変化しないも許容
                    if state.actions.len() + mx * 3 + 2 <= max_action && score <= best_score {
                        best_score = score;
                        best_diff = diff;
                        best_nxt1 = *nxt1;
                        best_nxt2 = *nxt2;
                        best_actions1 = actions1.to_vec();
                        best_actions2 = actions2.to_vec();
                    }
                }
            }
            // 改善する組合せがない場合は終了
            if best_actions1.is_empty() && best_actions2.is_empty() {
                break;
            }

            state.cost += best_diff;
            state.pos1 = best_nxt1;
            state.pos2 = best_nxt2;
            state.swap(state.pos1, state.pos2);

            // アクションが短い方は待つ"."を追加
            while best_actions1.len() < best_actions2.len() {
                best_actions1.push(DIRS_MAP[&'.']);
            }
            while best_actions2.len() < best_actions1.len() {
                best_actions2.push(DIRS_MAP[&'.']);
            }
            for (act1, act2) in best_actions1.iter().zip(&best_actions2) {
                state.actions.push(*act1);
                state.actions.push(*act2);
                state.actions.push(0);
            }
            // 移動後にスワップ
            let L = state.actions.len();
            state.actions[L - 1] = 1;
        }
        // ここまでのアクションはスワップで終わっているおり、この状態では不正なので、移動を追加
        // 最後のアクションは何もしない
        state.actions.push(DIRS_MAP[&'.']);
        state.actions.push(DIRS_MAP[&'.']);

        if state.cost < best_state.cost {
            best_state = state;
        }
    }

    // アクションが残っている場合はランダムウォークで改善
    while !best_state.is_done() {
        let diff = best_state.swap_with_calc_cost_diff(
            best_state.pos1,
            best_state.pos2,
            &input.legal_actions,
        );
        let mut is_swap = 1;
        if diff < 0 {
            best_state.swap(best_state.pos1, best_state.pos2);
            best_state.cost += diff;
        } else {
            is_swap = 0;
        }
        let idx1 = rng.gen_range(0..input.legal_actions[best_state.pos1].len());
        let idx2 = rng.gen_range(0..input.legal_actions[best_state.pos2].len());
        let (dir1, next1) = input.legal_actions[best_state.pos1][idx1];
        let (dir2, next2) = input.legal_actions[best_state.pos2][idx2];
        best_state.actions.push(is_swap);
        best_state.actions.push(dir1);
        best_state.actions.push(dir2);
        best_state.pos1 = next1;
        best_state.pos2 = next2;
    }

    best_state.output();
    eprintln!("Cost: {}", best_state.cost);
    eprintln!("Score: {}", best_state.calc_score(input.cost));
    eprintln!("Count: {}", cnt);
}

fn init_pos(n: usize, rng: &mut rand_pcg::Pcg64Mcg) -> Coord {
    let i = rng.gen_range(0..n);
    let j = rng.gen_range(0..n);
    Coord { row: i, col: j }
}

fn bfs(pos: Coord, input: &Input, dist_max: usize) -> Vec<(usize, Coord, Vec<usize>)> {
    let mut dist: FxHashMap<Coord, usize> = FxHashMap::default();
    dist.insert(pos, 0);
    let mut Q = VecDeque::new();
    let mut ret = vec![];
    Q.push_back((pos, vec![]));
    while let Some((pos, actions)) = Q.pop_front() {
        ret.push((dist[&pos], pos, actions.clone()));
        if dist[&pos] >= dist_max {
            continue;
        }
        for &(dir, nxt) in input.legal_actions[pos].iter() {
            if !dist.contains_key(&nxt) || dist[&pos] + 1 < dist[&nxt] {
                let mut nxt_actions = actions.clone();
                nxt_actions.push(dir);
                dist.insert(nxt, dist[&pos] + 1);
                Q.push_back((nxt, nxt_actions));
            }
        }
    }
    ret
}

const DIRS: [char; 5] = ['U', 'D', 'L', 'R', '.'];
const DIJ: [(usize, usize); 5] = [(!0, 0), (1, 0), (0, !0), (0, 1), (0, 0)];

lazy_static::lazy_static! {
    static ref DIRS_MAP: HashMap<char, usize> = {
        let mut mp = HashMap::new();
        for (i,dir) in DIRS.iter().enumerate() {
            mp.insert(*dir, i);
        }
        mp
    };
}

struct State {
    n: usize,
    init_pos1: Coord,
    init_pos2: Coord,
    pos1: Coord,
    pos2: Coord,
    cost: isize,
    board: DynamicMap2d<isize>,
    actions: Vec<usize>,
}

impl State {
    fn default() -> Self {
        State {
            n: 1,
            init_pos1: Coord { row: 0, col: 0 },
            init_pos2: Coord { row: 0, col: 0 },
            pos1: Coord { row: 0, col: 0 },
            pos2: Coord { row: 0, col: 0 },
            cost: std::isize::MAX,
            board: DynamicMap2d::new_with(0, 1),
            actions: vec![],
        }
    }
    fn new(n: usize, cost: isize, pos1: Coord, pos2: Coord, board: DynamicMap2d<isize>) -> Self {
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
    #[fastout]
    fn output(&mut self) {
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
    fn calc_score(&self, init_cost: isize) -> usize {
        let score = (init_cost as f64).log2() - (self.cost as f64).log2();
        let mut score = (1e6 * score).round() as usize;
        score = score.max(1);
        score
    }
    fn is_done(&self) -> bool {
        self.actions.len() >= 4 * self.n * self.n * 3
    }
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
            legal_actions[coord].push((DIRS_MAP[&'.'], coord));

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
