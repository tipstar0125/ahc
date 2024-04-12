#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::neg_multiply)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

use amplify::confinement::Collection;
use bitset_fixed::BitSet;
use itertools::Itertools;
use proconio::{input, marker::Usize1};
use rand::prelude::*;
use rustc_hash::FxHashSet;
use superslice::*;

const MOD: usize = 998244353;

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
    let input = read_input();
    let mut state = State::new(&input);
    let mut best_state = state.clone();

    while !state.is_done() {
        let mut cands = vec![];
        for m in 0..input.M {
            for i in 0..input.N - 2 {
                for j in 0..input.N - 2 {
                    let mut next_state = state.clone();
                    let pos = Coord::new(i, j);
                    next_state.apply(pos, &input.S[m]);
                    next_state.greedy(&input);
                    cands.push((next_state.score, m, i, j));
                }
            }
        }
        cands.sort();
        cands.reverse();
        let (_, m, i, j) = cands[0];
        let pos = Coord::new(i, j);
        state.apply(pos, &input.S[m]);
        state.actions.push((m, i, j));
        state.greedy(&input);
        if state.score > best_state.score {
            best_state = state.clone();
        }
    }

    best_state.output();
    eprintln!("Score: {}", best_state.score);
}

fn annealing(
    mut state: State,
    input: &Input,
    rng: &mut rand_pcg::Pcg64Mcg,
    time_keeper: &TimeKeeper,
    time_limit: f64,
) -> State {
    let T0 = (4 * MOD) as f64;
    let T1 = 1.0;
    let mut best_state = state.clone();
    let mut iter = 0;
    while time_keeper.get_time() < time_limit {
        iter += 1;
        let coin = rng.gen_range(0..10);
        if coin == 0 {
            if state.actions.is_empty() {
                continue;
            }
            let idx = rng.gen_range(0..state.actions.len());
            let (m, i, j) = state.actions[idx];
            let pos = Coord::new(i, j);
            let diff = state.calc_diff_score_revert(pos, &input.S[m]);
            let temp = (T0 + (T1 - T0) * time_keeper.get_time() / time_limit).max(T1);
            if diff >= 0 || rng.gen_bool((diff as f64 / temp).exp().min(1.0)) {
                state.revert(pos, &input.S[m]);
                state.actions.remove(idx);
            }
            if state.score > best_state.score {
                eprintln!("removed");
                best_state = state.clone();
            }
        } else if coin == 1 {
            if state.is_done() {
                continue;
            }
            let m = rng.gen_range(0..input.M);
            let i = rng.gen_range(0..input.N - 2);
            let j = rng.gen_range(0..input.N - 2);
            let pos = Coord::new(i, j);
            let diff = state.calc_diff_score(pos, &input.S[m]);
            let temp = T0 + (T1 - T0) * time_keeper.get_time() / time_limit;
            if diff >= 0 || rng.gen_bool((diff as f64 / temp).exp().min(1.0)) {
                state.apply(pos, &input.S[m]);
                state.actions.push((m, i, j));
            }
            if state.score > best_state.score {
                eprintln!("added");
                best_state = state.clone();
            }
        } else if coin <= 5 {
            if state.actions.is_empty() {
                continue;
            }
            let idx = rng.gen_range(0..state.actions.len());
            let (m0, i, j) = state.actions[idx];
            let m1 = rng.gen_range(0..input.M);
            let pos = Coord::new(i, j);
            let diff0 = state.calc_diff_score_revert(pos, &input.S[m0]);
            let diff1 = state.calc_diff_score(pos, &input.S[m1]);
            let diff = diff0 + diff1;
            let temp = T0 + (T1 - T0) * time_keeper.get_time() / time_limit;
            if diff >= 0 || rng.gen_bool((diff as f64 / temp).exp().min(1.0)) {
                state.revert(pos, &input.S[m0]);
                state.actions.remove(idx);
                state.apply(pos, &input.S[m1]);
                state.actions.push((m1, i, j));
            }
            if state.score > best_state.score {
                eprintln!("changed");
                best_state = state.clone();
            }
        } else {
            if state.actions.is_empty() {
                continue;
            }
            let idx = rng.gen_range(0..state.actions.len());
            let (m, i, j) = state.actions[idx];
            let pos = Coord::new(i, j);
            let diff0 = state.calc_diff_score_revert(pos, &input.S[m]);
            let nxt = pos + ADJ[rng.gen_range(0..4)];
            if nxt.row >= input.N - 2 || nxt.col >= input.N - 2 {
                continue;
            }
            let diff1 = state.calc_diff_score(nxt, &input.S[m]);
            let diff = diff0 + diff1;
            let temp = T0 + (T1 - T0) * time_keeper.get_time() / time_limit;
            if diff >= 0 || rng.gen_bool((diff as f64 / temp).exp().min(1.0)) {
                state.revert(pos, &input.S[m]);
                state.actions.remove(idx);
                state.apply(nxt, &input.S[m]);
                state.actions.push((m, nxt.row, nxt.col));
            }
            if state.score > best_state.score {
                eprintln!("moved");
                best_state = state.clone();
            }
        }
    }
    eprintln!("Iter: {}", iter);
    best_state
}

#[derive(Clone)]
struct State {
    A: DynamicMap2d<Mod>,
    K: usize,
    score: i64,
    turn: usize,
    actions: Vec<(usize, usize, usize)>,
}

impl State {
    fn new(input: &Input) -> State {
        let mut score = 0;
        for i in 0..input.N {
            for j in 0..input.N {
                let pos = Coord::new(i, j);
                score += input.A[pos].value() as i64;
            }
        }
        State {
            A: input.A.clone(),
            K: input.K,
            score,
            turn: 0,
            actions: vec![],
        }
    }
    fn calc_diff_score(&self, pos: Coord, s: &[[Mod; 3]; 3]) -> i64 {
        let mut before = 0;
        let mut after = 0;
        for i in 0..3 {
            for j in 0..3 {
                let coord_diff = CoordDiff::new(i as isize, j as isize);
                let nxt = pos + coord_diff;
                before += self.A[nxt].value() as i64;
                after += (self.A[nxt] + s[i][j]).value() as i64;
            }
        }
        after - before
    }
    fn calc_diff_score_revert(&self, pos: Coord, s: &[[Mod; 3]; 3]) -> i64 {
        let mut before = 0;
        let mut after = 0;
        for i in 0..3 {
            for j in 0..3 {
                let coord_diff = CoordDiff::new(i as isize, j as isize);
                let nxt = pos + coord_diff;
                before += self.A[nxt].value() as i64;
                after += (self.A[nxt] - s[i][j]).value() as i64;
            }
        }
        after - before
    }
    fn apply(&mut self, pos: Coord, s: &[[Mod; 3]; 3]) {
        let mut before = 0;
        let mut after = 0;
        for i in 0..3 {
            for j in 0..3 {
                let coord_diff = CoordDiff::new(i as isize, j as isize);
                let nxt = pos + coord_diff;
                before += self.A[nxt].value() as i64;
                after += (self.A[nxt] + s[i][j]).value() as i64;
                self.A[nxt] += s[i][j];
            }
        }
        self.turn += 1;
        self.score += after - before;
    }
    fn revert(&mut self, pos: Coord, s: &[[Mod; 3]; 3]) {
        let mut before = 0;
        let mut after = 0;
        for i in 0..3 {
            for j in 0..3 {
                let coord_diff = CoordDiff::new(i as isize, j as isize);
                let nxt = pos + coord_diff;
                before += self.A[nxt].value() as i64;
                after += (self.A[nxt] - s[i][j]).value() as i64;
                self.A[nxt] -= s[i][j];
            }
        }
        self.turn -= 1;
        self.score += after - before;
    }
    fn greedy(&mut self, input: &Input) {
        while !self.is_done() {
            let mut cands = vec![];
            for m in 0..input.M {
                for i in 0..input.N - 2 {
                    for j in 0..input.N - 2 {
                        let pos = Coord::new(i, j);
                        let diff = self.calc_diff_score(pos, &input.S[m]);
                        if diff >= 0 {
                            cands.push((diff, m, i, j));
                        }
                    }
                }
            }
            if cands.is_empty() {
                break;
            }
            cands.sort();
            cands.reverse();
            let (_, m, i, j) = cands[0];
            self.actions.push((m, i, j));
            self.apply(Coord::new(i, j), &input.S[m]);
        }
    }
    fn is_done(&self) -> bool {
        self.turn >= self.K
    }
    fn output(&self) {
        println!("{}", self.actions.len());
        for row in &self.actions {
            println!("{} {} {}", row.0, row.1, row.2);
        }
    }
}

struct Input {
    N: usize,
    M: usize,
    K: usize,
    A: DynamicMap2d<Mod>,
    S: Vec<[[Mod; 3]; 3]>,
}

fn read_input() -> Input {
    input! {
        N: usize,
        M: usize,
        K: usize,
        A2: [[usize; N]; N],
    }

    let mut A = DynamicMap2d::new_with(Mod::zero(), N);
    for i in 0..N {
        for j in 0..N {
            let pos = Coord::new(i, j);
            A[pos] = Mod::new(A2[i][j]);
        }
    }

    let mut S = vec![];
    for _ in 0..M {
        input! {
            s2: [[usize; 3]; 3]
        }
        let mut s1 = [[Mod::zero(); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                s1[i][j] = Mod::new(s2[i][j]);
            }
        }
        S.push(s1);
    }
    Input { N, M, K, A, S }
}

pub trait ChangeMinMax {
    fn chmin(&mut self, x: Self) -> bool;
    fn chmax(&mut self, x: Self) -> bool;
}

impl<T: PartialOrd> ChangeMinMax for T {
    fn chmin(&mut self, x: Self) -> bool {
        *self > x && {
            *self = x;
            true
        }
    }
    fn chmax(&mut self, x: Self) -> bool {
        *self < x && {
            *self = x;
            true
        }
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
    CoordDiff::new(0, 1),
    CoordDiff::new(0, -1),
    CoordDiff::new(1, 0),
    CoordDiff::new(-1, 0),
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

type Mod = ModInt;
#[derive(Debug, Clone, Copy, Default)]
struct ModInt {
    value: usize,
}

impl ModInt {
    fn new(n: usize) -> Self {
        ModInt { value: n % MOD }
    }
    fn zero() -> Self {
        ModInt { value: 0 }
    }
    fn one() -> Self {
        ModInt { value: 1 }
    }
    fn value(&self) -> usize {
        self.value
    }
    fn pow(&self, n: usize) -> Self {
        let mut p = *self;
        let mut ret = ModInt::one();
        let mut nn = n;
        while nn > 0 {
            if nn & 1 == 1 {
                ret *= p;
            }
            p *= p;
            nn >>= 1;
        }
        ret
    }
    fn inv(&self) -> Self {
        fn ext_gcd(a: usize, b: usize) -> (isize, isize, usize) {
            if a == 0 {
                return (0, 1, b);
            }
            let (x, y, g) = ext_gcd(b % a, a);
            (y - b as isize / a as isize * x, x, g)
        }

        ModInt::new((ext_gcd(self.value, MOD).0 + MOD as isize) as usize)
    }
}

impl std::ops::Add for ModInt {
    type Output = ModInt;
    fn add(self, other: Self) -> Self {
        ModInt::new(self.value + other.value)
    }
}

impl std::ops::Sub for ModInt {
    type Output = ModInt;
    fn sub(self, other: Self) -> Self {
        ModInt::new(MOD + self.value - other.value)
    }
}

impl std::ops::Mul for ModInt {
    type Output = ModInt;
    fn mul(self, other: Self) -> Self {
        ModInt::new(self.value * other.value)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl std::ops::Div for ModInt {
    type Output = ModInt;
    fn div(self, other: Self) -> Self {
        self * other.inv()
    }
}

impl std::ops::AddAssign for ModInt {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl std::ops::SubAssign for ModInt {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl std::ops::MulAssign for ModInt {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl std::ops::DivAssign for ModInt {
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}
