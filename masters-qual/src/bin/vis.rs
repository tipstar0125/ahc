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
    let input = Input {};
    let output = Output {};
    let max_turn = 1000;
    // #[cfg(feature = "vis")]
    {
        visualizer::vis(input, output, max_turn);
    }
}

pub struct Input {}
pub struct Output {}

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

            let N = 10;
            let d = VIS_WIDTH / N as f32;

            CentralPanel::default().show(ctx, |ui| {
                let hover_pos = ui.input().pointer.hover_pos();
                for i in 0..N {
                    for j in 0..N {
                        let pos1 = Pos2 {
                            y: i as f32 * d,
                            x: j as f32 * d,
                        };
                        let pos2 = Pos2 {
                            y: pos1.y + d,
                            x: pos1.x + d,
                        };
                        let rect = rect(ui, pos1, pos2, Color32::GRAY, Color32::WHITE);
                        let pos = Pos2 {
                            y: i as f32 * d + d / 2.0,
                            x: j as f32 * d + d / 2.0,
                        };
                        txt(ui, &(i * N + j).to_string(), pos, d / 3.0, Color32::BLACK);
                        if let Some(hover_pos) = hover_pos {
                            if rect.contains(hover_pos) {
                                show_tooltip_at_pointer(ui.ctx(), Id::new("hover tooltip"), |ui| {
                                    ui.label(format!("a[{}, {}] = {}", i, j, i * N + j));
                                });
                            }
                        }
                    }
                }
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
    // 0 <= opacity <= 1
    pub fn opacity(color: Color32, opacity: f32) -> Color32 {
        let opacity = (opacity * 255.0) as u8;
        Color32::from_rgba_premultiplied(color.r(), color.g(), color.b(), opacity)
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
