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
    let mut rng = rand_pcg::Pcg64Mcg::new(12345);
    let L = input.to.len();
    let time_limit = 2.0;
    let mut g = vec![vec![0; L + 2]; L + 2];

    for i in 0..L {
        for j in 0..L {
            g[i + 2][j + 2] = input.dist[input.to[i].0 * input.W + input.to[i].1]
                [input.to[j].0 * input.W + input.to[j].1];
        }
        g[i + 2][0] =
            input.dist[input.to[i].0 * input.W + input.to[i].1][input.st.0 * input.W + input.st.1];
        g[0][i + 2] = g[i + 2][0];
        g[i + 2][1] = 100000;
        g[1][i + 2] = 100000;
    }
    let greedy = tsp::greedy(&g);
    let mut route = tsp::solve(&g, &greedy, get_time() + time_limit, &mut rng);
    if route[1] == 1 {
        route.reverse();
    }
    let route = route[1..route.len() - 2]
        .iter()
        .map(|&v| input.to[v - 2])
        .collect_vec();

    let mut now = input.st;
    let mut actions = vec![];
    for &nxt in route.iter() {
        let ret = bfs(&input, now, nxt);
        actions.extend(ret);
        now = nxt;
    }
    let ans = actions.iter().join("");
    eprintln!("to length: {}", input.to.len());
    eprintln!("length: {}", ans.len());
    println!("{}", ans);
    let max_turn = actions.len();
    let output = Output { actions };
    visualizer::vis(input, output, max_turn);
}

fn bfs(input: &Input, start: (usize, usize), goal: (usize, usize)) -> Vec<char> {
    let mut Q = VecDeque::new();
    let mut dist = vec![vec![i32::max_value(); input.W]; input.H];
    dist[start.0][start.1] = 0;
    Q.push_back(start);
    while let Some(u) = Q.pop_front() {
        let d = dist[u.0][u.1];
        if u == goal {
            break;
        }
        for dir in 0..4 {
            if can_move(input, u.0, u.1, dir) {
                let v = (u.0 + DIJ[dir].0, u.1 + DIJ[dir].1);
                if dist[v.0][v.1].setmin(d + 1) {
                    Q.push_back(v);
                }
            }
        }
    }
    let mut now = goal;
    let mut ret = vec![];
    'a: while now != start {
        for dir in 0..4 {
            if can_move(input, now.0, now.1, dir) {
                let nxt = (now.0 + DIJ[dir].0, now.1 + DIJ[dir].1);
                if dist[now.0][now.1] == dist[nxt.0][nxt.1] + 1 {
                    now = nxt;
                    ret.push(REVERSE_DIR[dir]);
                    continue 'a;
                }
            }
        }
    }
    ret.reverse();
    ret
}

#[derive(Clone, Debug)]
pub struct Input {
    H: usize,
    W: usize,
    size: usize,
    st: (usize, usize),
    grid: Vec<Vec<char>>,
    to: Vec<(usize, usize)>,
    dist: Vec<Vec<i32>>,
}

fn read_input() -> Input {
    input! {
        H: usize,
        grid: [Chars; H]
    }

    let W = grid[0].len();
    let mut st = (0, 0);
    let mut to = vec![];

    for i in 0..H {
        for j in 0..W {
            if grid[i][j] == 'L' {
                st = (i, j);
            }
            if grid[i][j] == '.' {
                to.push((i, j));
            }
        }
    }
    let size = H * W;
    let mut input = Input {
        H,
        W,
        size,
        st,
        grid,
        to,
        dist: vec![],
    };

    let mut dist = mat![i32::max_value(); size; size];
    for s in 0..size {
        if input.grid[s / input.W][s % input.W] == '#' {
            continue;
        }
        dist[s][s] = 0;
        let mut Q = VecDeque::new();
        Q.push_back(s);
        while let Some(u) = Q.pop_front() {
            let d = dist[s][u];
            for dir in 0..4 {
                if can_move(&input, u / input.W, u % input.W, dir) {
                    let v = u + DIJ[dir].0 * input.W + DIJ[dir].1;
                    if dist[s][v].setmin(d + 1) {
                        Q.push_back(v);
                    }
                }
            }
        }
    }
    input.dist = dist;
    input
}

#[derive(Clone, Debug)]
pub struct Output {
    actions: Vec<char>,
}

const DIJ: [(usize, usize); 4] = [(!0, 0), (1, 0), (0, !0), (0, 1)];
const REVERSE_DIR: [char; 4] = ['D', 'U', 'R', 'L'];
const DIR_MAP: [char; 4] = ['U', 'D', 'L', 'R'];

fn can_move(input: &Input, i: usize, j: usize, dir: usize) -> bool {
    let (di, dj) = DIJ[dir];
    let i2 = i + di;
    let j2 = j + dj;
    if i2 >= input.H || j2 >= input.W {
        return false;
    }
    if input.grid[i2][j2] == '#' {
        return false;
    }
    true
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

mod visualizer {
    use crate::{Coord, Input, Output, DIJ, DIR_MAP};
    use eframe::egui::{
        show_tooltip_at_pointer, widgets, Align2, CentralPanel, Color32, Context, FontFamily,
        FontId, Id, Key, Pos2, Rect, RichText, Slider, Stroke, TextEdit, Ui,
    };
    use eframe::{run_native, App, CreationContext, Frame, NativeOptions, Storage, Theme};
    use std::time::{Duration, Instant};

    const WIDTH: f32 = 800.0;
    const HEIGHT: f32 = 800.0;
    const VIS_WIDTH: f32 = 600.0;
    const VIS_HEIGHT: f32 = 600.0;
    const OFFSET_WIDTH: f32 = (WIDTH - VIS_WIDTH) / 2.0;
    const OFFSET_HEIGHT: f32 = (HEIGHT - VIS_HEIGHT) / 2.0;
    const SPEED_MIN: usize = 1;
    const SPEED_MAX: usize = 10;
    const COLORS: [Color32; 7] = [
        Color32::WHITE,
        Color32::LIGHT_BLUE,
        Color32::BLUE,
        Color32::DARK_BLUE,
        Color32::LIGHT_GREEN,
        Color32::GREEN,
        Color32::DARK_GREEN,
    ];

    pub struct Egui {
        input: Input,
        output: Output,
        turn: usize,
        max_turn: usize,
        checked: bool,
        play: bool,
        speed: usize,
        instant: Instant,
        cnt: usize,
    }

    impl Egui {
        fn new(input: Input, output: Output, max_turn: usize) -> Self {
            Egui {
                input,
                output,
                turn: 0,
                max_turn,
                checked: true,
                play: false,
                speed: 5,
                instant: Instant::now(),
                cnt: 0,
            }
        }
    }

    impl App for Egui {
        fn save(&mut self, _storage: &mut dyn Storage) {}
        fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
            ctx.request_repaint_after(Duration::from_millis(5));
            if self.instant.elapsed() >= Duration::from_millis(10) {
                self.cnt += 1;
                if self.cnt % (SPEED_MIN + SPEED_MAX - self.speed) == 0
                    && self.play
                    && self.turn < self.max_turn
                {
                    self.turn += 1;
                }
                self.instant = Instant::now();
            }

            CentralPanel::default().show(ctx, |ui| {
                let H = self.input.H;
                let W = self.input.W;
                let d = VIS_WIDTH / H.max(W) as f32;
                let grid = self.input.grid.clone();
                let mut board = vec![vec![0; W]; H];
                let (mut x, mut y) = self.input.st;
                board[x][y] = 1;

                if self.turn > 0 {
                    for t in 0..self.turn {
                        let action = self.output.actions[t];
                        let action_num = {
                            let mut num = 0;
                            for (i, &dir) in DIR_MAP.iter().enumerate() {
                                if action == dir {
                                    num = i;
                                }
                            }
                            num
                        };
                        x += DIJ[action_num].0;
                        y += DIJ[action_num].1;
                        board[x][y] += 1;
                    }
                }
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Turn: ").size(20.0));
                    ui.add(Slider::new(&mut self.turn, 0..=self.max_turn));
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Speed: ").size(20.0));
                    ui.add(Slider::new(&mut self.speed, SPEED_MIN..=SPEED_MAX));
                });

                if ctx.input().key_released(Key::Space) {
                    self.play = !self.play;
                };
                if self.turn == self.max_turn {
                    self.play = false;
                }
                if ctx.input().key_pressed(Key::ArrowRight) && self.turn < self.max_turn {
                    self.turn += 1;
                };
                if ctx.input().key_pressed(Key::ArrowLeft) && self.turn > 0 {
                    self.turn -= 1;
                };

                // let hover_pos = ui.input().pointer.hover_pos();
                for i in 0..H {
                    for j in 0..W {
                        let pos1 = Pos2 {
                            x: j as f32 * d,
                            y: i as f32 * d,
                        };
                        let pos2 = Pos2 {
                            x: pos1.x + d,
                            y: pos1.y + d,
                        };
                        if grid[i][j] == '#' {
                            rect(ui, pos1, pos2, Color32::BLACK, Color32::WHITE);
                        } else {
                            let cnt = board[i][j];
                            rect(ui, pos1, pos2, COLORS[cnt], Color32::BLACK);
                        }
                        // if let Some(hover_pos) = hover_pos {
                        //     if rect.contains(hover_pos) {
                        //         show_tooltip_at_pointer(ui.ctx(), Id::new("hover tooltip"), |ui| {
                        //             ui.label(format!("a[{}, {}] = {}", i, j, board[coord]));
                        //         });
                        //     }
                        // }
                    }
                }
                let pos = Pos2 {
                    x: y as f32 * d + d / 2.0,
                    y: x as f32 * d + d / 2.0,
                };
                circle(ui, pos, d / 3.0, Color32::RED, Color32::WHITE);
            });
        }
    }

    pub fn vis(input: Input, output: Output, max_turn: usize) {
        let options = NativeOptions {
            initial_window_size: Some((WIDTH, HEIGHT).into()),
            initial_window_pos: Some(Pos2 { x: 100.0, y: 100.0 }),
            resizable: false,
            default_theme: Theme::Light,
            ..NativeOptions::default()
        };
        let gui = Egui::new(input, output, max_turn);
        run_native("visualizer", options, Box::new(|_cc| Box::new(gui)));
    }

    // 0 <= val <= 1
    pub fn color32(mut val: f32) -> Color32 {
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
        Color32::from_rgb(r.round() as u8, g.round() as u8, b.round() as u8)
    }
    pub fn txt(ui: &mut Ui, txt: &str, mut pos: Pos2, size: f32, color: Color32) {
        pos.x += OFFSET_WIDTH;
        pos.y += OFFSET_HEIGHT;
        let anchor = Align2::CENTER_CENTER;
        let font_id = FontId::new(size, FontFamily::Monospace);
        ui.painter().text(pos, anchor, txt, font_id, color);
    }
    pub fn line(ui: &mut Ui, mut pos1: Pos2, mut pos2: Pos2, color: Color32) {
        pos1.x += OFFSET_WIDTH;
        pos2.x += OFFSET_WIDTH;
        pos1.y += OFFSET_HEIGHT;
        pos2.y += OFFSET_HEIGHT;
        let points = [pos1, pos2];
        let stroke = Stroke { width: 3.0, color };
        ui.painter().line_segment(points, stroke);
    }
    #[allow(clippy::too_many_arguments)]
    pub fn dash_line(
        ui: &mut Ui,
        mut pos1: Pos2,
        mut pos2: Pos2,
        dash_length: f32,
        gap_length: f32,
        color: Color32,
    ) {
        pos1.x += OFFSET_WIDTH;
        pos2.x += OFFSET_WIDTH;
        pos1.y += OFFSET_HEIGHT;
        pos2.y += OFFSET_HEIGHT;

        let mut x1 = pos1.x;
        let mut y1 = pos1.y;
        let mut x2 = pos2.x;
        let mut y2 = pos2.y;
        let mut is_swap = false;
        if x1 == x2 {
            std::mem::swap(&mut x1, &mut y1);
            std::mem::swap(&mut x2, &mut y2);
            is_swap = true;
        }

        let a = (y2 - y1) / (x2 - x1);
        let b = y1 - a * x1;

        let f = |x: f32, target_length: f32| -> bool {
            let y = a * x + b;
            let len = ((x - x1).powf(2.0) + (y - y1).powf(2.0)).sqrt();
            len <= target_length
        };

        let mut positions = vec![];
        positions.push((x1, y1));
        let max_length = ((x1 - x2).powf(2.0) + (y1 - y2).powf(2.0)).sqrt();
        let mut cnt = 0;
        let mut target_length = 0.0f32;

        loop {
            if cnt % 2 == 0 {
                target_length += dash_length;
            } else {
                target_length += gap_length;
            }
            if target_length >= max_length {
                break;
            }

            let mut ok = x1;
            let mut ng = x2;
            while (ng - ok).abs() > 1e-3 {
                let m = (ok + ng) / 2.0;
                if f(m, target_length) {
                    ok = m;
                } else {
                    ng = m;
                }
            }

            positions.push((ok, a * ok + b));

            cnt += 1;
        }
        positions.push((x2, y2));
        if positions.len() % 2 == 1 {
            positions.pop();
        }
        if is_swap {
            for (a, b) in &mut positions {
                std::mem::swap(a, b);
            }
        }
        let mut i = 0;
        while i < positions.len() {
            let p1 = Pos2 {
                x: positions[i].0,
                y: positions[i].1,
            };
            let p2 = Pos2 {
                x: positions[i + 1].0,
                y: positions[i + 1].1,
            };
            line(ui, p1, p2, color);
            i += 2;
        }
    }
    pub fn rect(
        ui: &mut Ui,
        mut pos1: Pos2,
        mut pos2: Pos2,
        fill_color: Color32,
        stroke_color: Color32,
    ) -> Rect {
        pos1.x += OFFSET_WIDTH;
        pos2.x += OFFSET_WIDTH;
        pos1.y += OFFSET_HEIGHT;
        pos2.y += OFFSET_HEIGHT;

        let rect = Rect {
            min: pos1,
            max: pos2,
        };
        let rounding = 0.0;
        let stroke = Stroke {
            width: 0.2,
            color: stroke_color,
        };
        ui.painter().rect(rect, rounding, fill_color, stroke);
        rect
    }
    pub fn circle(
        ui: &mut Ui,
        mut center: Pos2,
        radius: f32,
        fill_color: Color32,
        stroke_color: Color32,
    ) {
        center.x += OFFSET_WIDTH;
        center.y += OFFSET_HEIGHT;
        let stroke = Stroke {
            width: 0.2,
            color: stroke_color,
        };
        ui.painter().circle(center, radius, fill_color, stroke);
    }
    pub fn partition(ui: &mut Ui, h: &[Vec<char>], v: &[Vec<char>], size: f32) {
        let H = v.len();
        let W = h[0].len();
        for i in 0..H + 1 {
            for j in 0..W {
                // Entrance
                // if i == 0 && j == ENTRANCE {
                //     continue;
                // }
                if (i == 0 || i == H) || h[i - 1][j] == '1' {
                    let pos1 = Pos2 {
                        x: j as f32 * size,
                        y: i as f32 * size,
                    };
                    let pos2 = Pos2 {
                        x: pos1.x + size,
                        y: pos1.y,
                    };
                    line(ui, pos1, pos2, Color32::BLACK);
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
                    let pos1 = Pos2 {
                        x: j as f32 * size,
                        y: i as f32 * size,
                    };
                    let pos2 = Pos2 {
                        x: pos1.x,
                        y: pos1.y + size,
                    };
                    line(ui, pos1, pos2, Color32::BLACK);
                }
            }
        }
    }
}
