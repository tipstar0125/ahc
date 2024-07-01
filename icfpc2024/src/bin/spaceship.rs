#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::neg_multiply)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

use std::collections::VecDeque;

use itertools::Itertools;
use proconio::{
    input,
    marker::{Chars, Usize1},
};
use rand::prelude::*;
use rustc_hash::FxHashSet;
use superslice::*;

fn main() {
    get_time();
    solve();
    eprintln!("Elapsed: {}", (get_time() * 1000.0) as usize);
}

fn solve() {
    let input = read_input();
    let mut cnt = 0;
    let mut action_cnt = 0;
    let mut visited = vec![false; input.N];
    let mut x = 0;
    let mut y = 0;
    let mut vx = 0;
    let mut vy = 0;

    while cnt < input.N {
        let mut cands = vec![];
        for (i, &vis) in visited.iter().enumerate() {
            if vis {
                continue;
            }
            let (nx, ny) = input.coords[i];
            let d = calc_dist(x, y, nx, ny);
            cands.push((d, i));
        }
        cands.sort();
        let (nx, ny) = input.coords[cands[0].1];
        let mut action = 5;
        if x + vx < nx {
            action += 1;
            vx += 1;
        }
        if x + vx > nx {
            action -= 1;
            vx -= 1;
        }
        if y + vy < ny {
            action += 3;
            vy += 1;
        }
        if y + vy > ny {
            action -= 3;
            vy -= 1;
        }
        x += vx;
        y += vy;
        print!("{}", action);
        // eprintln!("{} {} {} {} {}", x, y, vx, vy, action);
        action_cnt += 1;

        for (i, &(nx, ny)) in input.coords.iter().enumerate() {
            if (x, y) == (nx, ny) && !visited[i] {
                visited[i] = true;
                cnt += 1;
            }
        }
    }
    // eprintln!("action cnt: {}", action_cnt);
}

fn calc_dist(x0: isize, y0: isize, x1: isize, y1: isize) -> isize {
    (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1)
}

#[derive(Clone, Debug)]
struct Input {
    N: usize,
    coords: Vec<(isize, isize)>,
}

fn read_input() -> Input {
    input! {
        n: usize,
        mut coords: [(isize, isize); n]
    }
    coords.sort();
    coords.dedup();
    let N = coords.len();
    Input { N, coords }
}

#[macro_export]
macro_rules! mat {
	($($e:expr),*) => { vec![$($e),*] };
	($($e:expr,)*) => { vec![$($e),*] };
	($e:expr; $d:expr) => { vec![$e; $d] };
	($e:expr; $d:expr $(; $ds:expr)+) => { vec![mat![$e $(; $ds)*]; $d] };
}

pub trait SetMinMax {
    fn setmin(&mut self, v: Self) -> bool;
    fn setmax(&mut self, v: Self) -> bool;
}
impl<T> SetMinMax for T
where
    T: PartialOrd,
{
    fn setmin(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }
    fn setmax(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

mod tsp {

    use super::*;
    use rand_pcg::Pcg64Mcg;
    type C = i32;

    pub fn compute_cost(g: &[Vec<C>], ps: &Vec<usize>) -> C {
        let mut tmp = 0;
        for i in 0..ps.len() - 1 {
            tmp += g[ps[i]][ps[i + 1]];
        }
        tmp
    }

    pub fn greedy(g: &Vec<Vec<C>>) -> Vec<usize> {
        let mut ps = vec![0];
        let n = g.len();
        let mut used = vec![false; n];
        used[0] = true;
        for i in 0..n - 1 {
            let mut to = !0;
            let mut cost = C::max_value();
            for j in 0..n {
                if !used[j] && cost.setmin(g[i][j]) {
                    to = j;
                }
            }
            used[to] = true;
            ps.push(to);
        }
        ps.push(0);
        ps
    }

    // mv: (i, dir)
    pub fn apply_move(tour: &mut [usize], idx: &mut [usize], mv: &[(usize, usize)]) {
        let k = mv.len();
        let mut ids: Vec<_> = (0..k).collect();
        ids.sort_by_key(|&i| mv[i].0);
        let mut order = vec![0; k];
        for i in 0..k {
            order[ids[i]] = i;
        }
        let mut tour2 = Vec::with_capacity(mv[ids[k - 1]].0 - mv[ids[0]].0);
        let mut i = ids[0];
        let mut dir = 0;
        loop {
            let (j, rev) = if dir == mv[i].1 {
                ((i + 1) % k, 0)
            } else {
                ((i + k - 1) % k, 1)
            };
            if mv[j].1 == rev {
                if order[j] == k - 1 {
                    break;
                } else {
                    i = ids[order[j] + 1];
                    dir = 0;
                    tour2.extend_from_slice(&tour[mv[j].0 + 1..mv[i].0 + 1]);
                }
            } else {
                i = ids[order[j] - 1];
                dir = 1;
                tour2.extend(tour[mv[i].0 + 1..mv[j].0 + 1].iter().rev().cloned());
            }
        }
        assert_eq!(tour2.len(), mv[ids[k - 1]].0 - mv[ids[0]].0);
        tour[mv[ids[0]].0 + 1..mv[ids[k - 1]].0 + 1].copy_from_slice(&tour2);
        for i in mv[ids[0]].0 + 1..mv[ids[k - 1]].0 + 1 {
            idx[tour[i]] = i;
        }
    }

    pub const FEASIBLE3: [bool; 64] = [
        false, false, false, true, false, true, true, true, true, true, true, false, true, false,
        false, false, false, false, false, false, false, false, false, false, false, false, false,
        true, false, true, true, true, true, true, true, false, true, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false, true, false, true,
        true, true, true, true, true, false, true, false, false, false,
    ];

    pub fn solve(g: &Vec<Vec<C>>, qs: &Vec<usize>, until: f64, rng: &mut Pcg64Mcg) -> Vec<usize> {
        let n = g.len();
        let mut f = vec![vec![]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    f[i].push((g[i][j], j));
                }
            }
            f[i].sort_by(|&(a, _), &(b, _)| a.partial_cmp(&b).unwrap());
        }
        let mut ps = qs.clone();
        let mut idx = vec![!0; n];
        let (mut min, mut min_ps) = (compute_cost(g, qs), ps.clone());
        while get_time() < until {
            let mut cost = compute_cost(g, &ps);
            for p in 0..n {
                idx[ps[p]] = p;
            }
            loop {
                let mut ok = false;
                for i in 0..n {
                    for di in 0..2 {
                        'loop_ij: for &(ij, vj) in &f[ps[i + di]] {
                            if g[ps[i]][ps[i + 1]] - ij <= 0 {
                                break;
                            }
                            for dj in 0..2 {
                                let j = if idx[vj] == 0 && dj == 0 {
                                    n - 1
                                } else {
                                    idx[vj] - 1 + dj
                                };
                                let gain = g[ps[i]][ps[i + 1]] - ij + g[ps[j]][ps[j + 1]];
                                // 2-opt
                                if di != dj && gain - g[ps[j + dj]][ps[i + 1 - di]] > 0 {
                                    cost -= gain - g[ps[j + dj]][ps[i + 1 - di]];
                                    apply_move(&mut ps, &mut idx, &[(i, di), (j, dj)]);
                                    ok = true;
                                    break 'loop_ij;
                                }
                                // 3-opt
                                for &(jk, vk) in &f[ps[j + dj]] {
                                    if gain - jk <= 0 {
                                        break;
                                    }
                                    for dk in 0..2 {
                                        let k = if idx[vk] == 0 && dk == 0 {
                                            n - 1
                                        } else {
                                            idx[vk] - 1 + dk
                                        };
                                        if i == k || j == k {
                                            continue;
                                        }
                                        let gain = gain - jk + g[ps[k]][ps[k + 1]];
                                        if gain - g[ps[k + dk]][ps[i + 1 - di]] > 0 {
                                            let mask = if i < j { 1 << 5 } else { 0 }
                                                | if i < k { 1 << 4 } else { 0 }
                                                | if j < k { 1 << 3 } else { 0 }
                                                | di << 2
                                                | dj << 1
                                                | dk;
                                            if FEASIBLE3[mask] {
                                                cost -= gain - g[ps[k + dk]][ps[i + 1 - di]];
                                                apply_move(
                                                    &mut ps,
                                                    &mut idx,
                                                    &[(i, di), (j, dj), (k, dk)],
                                                );
                                                ok = true;
                                                break 'loop_ij;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if !ok {
                    break;
                }
            }
            if min.setmin(cost) {
                min_ps = ps;
            }
            ps = min_ps.clone();
            if n <= 4 {
                break;
            }
            loop {
                if rng.gen_range(0..2) == 0 {
                    // double bridge
                    let mut is: Vec<_> = (0..4).map(|_| rng.gen_range(0..n)).collect();
                    is.sort();
                    if is[0] == is[1] || is[1] == is[2] || is[2] == is[3] {
                        continue;
                    }
                    ps = ps[0..is[0] + 1]
                        .iter()
                        .chain(ps[is[2] + 1..is[3] + 1].iter())
                        .chain(ps[is[1] + 1..is[2] + 1].iter())
                        .chain(ps[is[0] + 1..is[1] + 1].iter())
                        .chain(ps[is[3] + 1..].iter())
                        .cloned()
                        .collect();
                } else {
                    for _ in 0..6 {
                        loop {
                            let i = rng.gen_range(1..n);
                            let j = rng.gen_range(1..n);
                            if i < j && j - i < n - 2 {
                                ps = ps[0..i]
                                    .iter()
                                    .chain(ps[i..j + 1].iter().rev())
                                    .chain(ps[j + 1..].iter())
                                    .cloned()
                                    .collect();
                                break;
                            }
                        }
                    }
                }
                break;
            }
        }
        min_ps
    }
}

pub fn get_time() -> f64 {
    static mut STIME: f64 = -1.0;
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    let ms = t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9;
    unsafe {
        if STIME < 0.0 {
            STIME = ms;
        }
        // ローカル環境とジャッジ環境の実行速度差はget_timeで吸収しておくと便利
        #[cfg(feature = "local")]
        {
            (ms - STIME) * 1.0
        }
        #[cfg(not(feature = "local"))]
        {
            ms - STIME
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
