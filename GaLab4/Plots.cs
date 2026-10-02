#if SCOTTPLOT
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using ScottPlot;

namespace GaLab4
{
    static class Plots
    {
        static double P(string s) { return double.Parse(s, CultureInfo.InvariantCulture); }

        public static void Make()
        {
            // --- Частина 2: збіжність на кожному наборі даних ---
            var rows = File.ReadAllLines("convergence_long.csv").Skip(1).Select(l => l.Split(',')).ToList();
            foreach (var ds in rows.Select(r => r[0]).Distinct())
            {
                var fit = new Plot();
                var feas = new Plot();
                foreach (var cfg in rows.Where(r => r[0] == ds).Select(r => r[1]).Distinct())
                {
                    var rs = rows.Where(r => r[0] == ds && r[1] == cfg).ToList();
                    double[] xs = rs.Select(r => P(r[3])).ToArray();                 
                    double[] yf = rs.Select(r => Math.Log10(1.0 + P(r[4]))).ToArray(); 
                    double[] yp = rs.Select(r => P(r[6])).ToArray();                 

                    var s1 = fit.Add.Scatter(xs, yf); s1.LegendText = cfg; s1.MarkerSize = 0; s1.LineWidth = 2;
                    var s2 = feas.Add.Scatter(xs, yp); s2.LegendText = cfg; s2.MarkerSize = 0; s2.LineWidth = 2;
                }
                fit.Title("Динаміка fitness, " + ds);
                fit.XLabel("Кількість оцінок fitness");
                fit.YLabel("log10(1 + fitness)");
                fit.ShowLegend();
                fit.SavePng("convergence_fitness_" + ds + ".png", 1000, 600);

                feas.Title("Частка допустимих особин (hard = 0), " + ds);
                feas.XLabel("Кількість оцінок fitness");
                feas.YLabel("Частка популяції");
                feas.ShowLegend();
                feas.SavePng("convergence_feasible_" + ds + ".png", 1000, 600);
            }

            // --- Частина 1: ГА проти градієнтного спуску за кількістю викликів f ---
            var p1 = File.ReadAllLines("part1_convergence.csv").Skip(1).Select(l => l.Split(',')).ToList();
            var plt = new Plot();
            double[] x1 = p1.Select(r => P(r[0])).ToArray();
            var ga = plt.Add.Scatter(x1, p1.Select(r => Math.Log10(P(r[2]) + 1e-6)).ToArray());
            ga.LegendText = "ГА (медіана)"; ga.LineWidth = 2;
            var gd = p1.Where(r => r[4] != "").ToList();
            var gds = plt.Add.Scatter(gd.Select(r => P(r[0])).ToArray(), gd.Select(r => Math.Log10(P(r[4]) + 1e-6)).ToArray());
            gds.LegendText = "GD з перезапусками (медіана)"; gds.LineWidth = 2;
            plt.Title("Задача 1: збіжність за бюджетом викликів f");
            plt.XLabel("Кількість викликів f");
            plt.YLabel("log10(найкраще f)");
            plt.ShowLegend();
            plt.SavePng("part1_convergence.png", 1000, 600);
        }
    }
}
#endif