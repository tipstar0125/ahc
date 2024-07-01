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
use proconio::{input_interactive, marker::Chars};
use rand::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

const INF: isize = 1_isize << 60;
const MAX_TURN: usize = 10;

fn main() {
    let start = std::time::Instant::now();

    let input = read_input();
    let mut state = State::new(&input);

    'a: loop {
        let nxt = state.decide_next();
        if let Some(nxt) = nxt {
            loop {
                state.measure_x(nxt);
                if state.is_done() {
                    break 'a;
                }
                state.measure_y(nxt);
                if state.is_done() {
                    break 'a;
                }
            }
        } else {
            break;
        }
    }
    eprintln!("{:?}", state.visited);

    #[allow(unused_mut, unused_assignments)]
    let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
    #[cfg(feature = "local")]
    {
        eprintln!("Local Mode");
        elapsed_time *= 0.55;
    }
    eprintln!("Elapsed: {}", (elapsed_time * 1000.0) as usize);

    // #[cfg(feature = "vis")]
    // {
    //     let output = { Output {} };
    //     let max_turn = 1;
    //     visualizer::vis(input, output, max_turn);
    // }
}

struct State {
    N: usize,
    x: isize,
    y: isize,
    vx: isize,
    vy: isize,
    turn: usize,
    score: i64,
    root: Vec<usize>,
    visited: FxHashSet<usize>,
}

impl State {
    fn new(input: &Input) -> Self {
        let mut best_dist = INF;
        let mut best_root = vec![];
        for p in (0..input.N).permutations(input.N) {
            let mut dist = (input.sx - input.P[p[0]].0).abs() + (input.sy - input.P[p[0]].1).abs();
            for i in 1..input.N {
                dist += (input.P[p[i]].0 - input.P[p[i - 1]].0).pow(2)
                    + (input.P[p[i]].1 - input.P[p[i - 1]].1).pow(2);
            }
            if dist < best_dist {
                best_dist = dist;
                best_root = p;
            }
        }

        State {
            N: input.N,
            x: input.sx,
            y: input.sy,
            vx: 0,
            vy: 0,
            turn: 0,
            score: 0,
            root: best_root,
            visited: FxHashSet::default(),
        }
    }
    fn decide_next(&self) -> Option<usize> {
        for id in self.root.iter() {
            if !self.visited.contains(id) {
                return Some(*id);
            }
        }
        None
    }
    fn move_(&mut self) {
        self.x += self.vx;
        self.y += self.vy;
    }
    fn measure_x(&mut self, nxt: usize) -> bool {
        self.turn += 1;
        println!("S 1 0");
        input_interactive! {
            p: isize,
            c: usize,
            h: usize,
            q: [usize; h]
        }
        self.x = 1e5 as isize - p;
        self.move_();
        self.score -= 2;
        if c == 1 {
            self.score -= 100;
        }
        let before = self.visited.len() as i64;
        for &qi in q.iter() {
            self.visited.insert(qi);
        }
        let after = self.visited.len() as i64;
        self.score += 1000 * (after - before);
        self.visited.contains(&nxt)
    }
    fn measure_y(&mut self, nxt: usize) -> bool {
        self.turn += 1;
        println!("S 0 1");
        input_interactive! {
            p: isize,
            c: usize,
            h: usize,
            q: [usize; h]
        }
        self.y = 1e5 as isize - p;
        self.move_();
        self.score -= 2;
        if c == 1 {
            self.score -= 100;
        }
        let before = self.visited.len() as i64;
        for &qi in q.iter() {
            self.visited.insert(qi);
        }
        let after = self.visited.len() as i64;
        self.score += 1000 * (after - before);
        self.visited.contains(&nxt)
    }
    fn calc_target_velocity(&self, nxt: usize, input: &Input) {
        let diff_x = input.P[nxt].0 - self.x;
        let diff_y = input.P[nxt].1 - self.y;
        
    }
    // fn accel_x(&mut self, v: isize, nxt: usize) -> bool {
    //     self.turn += 1;
    //     self.velocity.0 += v;
    //     println!("A {} 0", v);
    //     input_interactive! {
    //         c: usize,
    //         h: usize,
    //         q: [usize; h]
    //     }
    //     self.score -= 2;
    //     if c == 1 {
    //         self.score -= 100;
    //     }
    //     let before = self.visited.len() as i64;
    //     for &qi in q.iter() {
    //         self.visited.insert(qi);
    //     }
    //     let after = self.visited.len() as i64;
    //     self.score += 1000 * (after - before);
    //     self.visited.contains(&nxt)
    // }
    // fn accel_y(&mut self, v: isize, nxt: usize) -> bool {
    //     self.turn += 1;
    //     self.velocity.1 += v;
    //     println!("A 0 {}", v);
    //     input_interactive! {
    //         c: usize,
    //         h: usize,
    //         q: [usize; h]
    //     }
    //     self.score -= 2;
    //     if c == 1 {
    //         self.score -= 100;
    //     }
    //     let before = self.visited.len() as i64;
    //     for &qi in q.iter() {
    //         self.visited.insert(qi);
    //     }
    //     let after = self.visited.len() as i64;
    //     self.score += 1000 * (after - before);
    //     self.visited.contains(&nxt)
    // }
    fn is_done(&self) -> bool {
        self.turn >= MAX_TURN || self.visited.len() == self.N
    }
}

pub struct Input {
    N: usize,
    M: usize,
    eps: f64,
    sigma: f64,
    sx: isize,
    sy: isize,
    P: Vec<(isize, isize)>,
    wall: Vec<(isize, isize, isize, isize)>,
}

pub struct Output {}

fn read_input() -> Input {
    input_interactive! {
        N: usize,
        M: usize,
        eps: f64,
        sigma: f64,
        sx: isize,
        sy: isize,
        P: [(isize, isize); N],
        wall: [(isize, isize, isize, isize); M],
    }

    Input {
        N,
        M,
        eps,
        sigma,
        sx,
        sy,
        P,
        wall,
    }
}

#[derive(Debug, Clone)]
pub struct TimeKeeper {
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
    pub fn get_time(&self) -> f64 {
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

// #[cfg(feature = "vis")]
mod visualizer {
    use crate::{Input, Output};
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
                turn: max_turn,
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
                let score = 0;
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("Score: {}", score)).size(20.0));
                    ui.checkbox(&mut self.checked, "Show Number");
                    widgets::global_dark_light_mode_buttons(ui);
                });
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

                let hover_pos = ui.input().pointer.hover_pos();
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
