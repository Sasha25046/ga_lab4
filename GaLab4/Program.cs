using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace GaLab4
{
    class Program
    {
    
        static Random GlobalRng = new Random(42);
        static void Seed(int s) { GlobalRng = new Random(s); }

        const double INF = double.PositiveInfinity;
        static readonly CultureInfo IC = CultureInfo.InvariantCulture;

        // =====================================================================
        // ЧАСТИНА 1: НЕГЛАДКА РОЗРИВНА ОПТИМІЗАЦІЯ
        // =====================================================================
        const int DIM = 5;
        static readonly double[] BOUNDS = { -5.12, 5.12 };
        static readonly double[] SHIFT = { 1.7, -2.3, 0.9, -1.4, 2.6 };
        const int BUDGET = 20000;
        const double SUCCESS_EPS = 0.1;
        const int GD_ITERS = 200;
        static readonly int GD_COST = GD_ITERS * (DIM + 1) + 1;

        static double ShiftedDiscontinuousRastrigin(double[] x)
        {
            double total = 10.0 * x.Length;
            for (int i = 0; i < x.Length; i++)
            {
                double z = x[i] - SHIFT[i];
                total += z * z - 10.0 * Math.Cos(2.0 * Math.PI * z) + Math.Floor(Math.Abs(z) * 2.0) * 1.5;
            }
            return total;
        }

        class CountedFunction
        {
            private readonly Func<double[], double> _func;
            public int Calls { get; private set; }

            public CountedFunction(Func<double[], double> func)
            {
                _func = func;
                Calls = 0;
            }

            public double Eval(double[] x)
            {
                Calls++;
                return _func(x);
            }
        }

        static (double[] point, double value) GradientDescent(CountedFunction func, int dim = DIM, double lr = 0.01, int iters = GD_ITERS, double eps = 1e-5)
        {
            double[] current = new double[dim];
            for (int i = 0; i < dim; i++)
                current[i] = BOUNDS[0] + GlobalRng.NextDouble() * (BOUNDS[1] - BOUNDS[0]);

            for (int it = 0; it < iters; it++)
            {
                double baseVal = func.Eval(current);
                double[] grad = new double[dim];
                for (int i = 0; i < dim; i++)
                {
                    double[] xPlus = (double[])current.Clone();
                    xPlus[i] += eps;
                    grad[i] = (func.Eval(xPlus) - baseVal) / eps;
                }

                for (int i = 0; i < dim; i++)
                {
                    double nextVal = current[i] - lr * grad[i];
                    current[i] = Math.Max(BOUNDS[0], Math.Min(BOUNDS[1], nextVal));
                }
            }
            return (current, func.Eval(current));
        }

        static (double[] point, double value) GradientDescentMultistart(CountedFunction func, int budget,
            double lr = 0.01, List<(int cost, double best)> trace = null)
        {
            (double[] point, double value) best = (null, INF);
            while (func.Calls + GD_COST <= budget)
            {
                var candidate = GradientDescent(func, DIM, lr);
                if (candidate.value < best.value)
                    best = candidate;
                if (trace != null) trace.Add((func.Calls, best.value));
            }
            return best;
        }

        static (double[] point, double value) ContinuousGa(
            CountedFunction func, int budget, int dim = DIM, int popSize = 50,
            double crossoverRate = 0.9, double mutationRate = 0.2, double blxAlpha = 0.5,
            double sigmaStart = 0.5, double sigmaEnd = 0.02, int eliteCount = 2, int tournamentSize = 3,
            List<double> history = null)
        {
            List<double[]> population = new List<double[]>();
            for (int i = 0; i < popSize; i++)
            {
                double[] ind = new double[dim];
                for (int d = 0; d < dim; d++)
                    ind[d] = BOUNDS[0] + GlobalRng.NextDouble() * (BOUNDS[1] - BOUNDS[0]);
                population.Add(ind);
            }

            int maxGen = budget / popSize;
            double[] bestInd = null;
            double bestScore = INF;

            double Clamp(double v) => Math.Max(BOUNDS[0], Math.Min(BOUNDS[1], v));

            double[] Select(List<double[]> pop, double[] scores)
            {
                int bestIdx = -1;
                double bestVal = INF;
                for (int t = 0; t < tournamentSize; t++)
                {
                    int r = GlobalRng.Next(popSize);
                    if (scores[r] < bestVal)
                    {
                        bestVal = scores[r];
                        bestIdx = r;
                    }
                }
                return pop[bestIdx];
            }

            double NextGaussian(double mean, double stdDev)
            {
                double u1 = 1.0 - GlobalRng.NextDouble();
                double u2 = 1.0 - GlobalRng.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                return mean + stdDev * randStdNormal;
            }

            for (int gen = 0; gen < maxGen; gen++)
            {
                double[] scores = new double[popSize];
                for (int i = 0; i < popSize; i++)
                    scores[i] = func.Eval(population[i]);

                int[] order = Enumerable.Range(0, popSize).OrderBy(i => scores[i]).ToArray();
                if (scores[order[0]] < bestScore)
                {
                    bestScore = scores[order[0]];
                    bestInd = (double[])population[order[0]].Clone();
                }
                if (history != null) history.Add(bestScore);

                double sigma = sigmaStart + (sigmaEnd - sigmaStart) * gen / Math.Max(1, maxGen - 1);
                List<double[]> newPopulation = new List<double[]>();
                for (int k = 0; k < eliteCount; k++)
                    newPopulation.Add((double[])population[order[k]].Clone());

                while (newPopulation.Count < popSize)
                {
                    double[] p1 = Select(population, scores);
                    double[] p2 = Select(population, scores);
                    double[] child = new double[dim];

                    if (GlobalRng.NextDouble() < crossoverRate)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            double lo = Math.Min(p1[d], p2[d]);
                            double hi = Math.Max(p1[d], p2[d]);
                            double range = hi - lo;
                            double minVal = lo - blxAlpha * range;
                            double maxVal = hi + blxAlpha * range;
                            child[d] = Clamp(minVal + GlobalRng.NextDouble() * (maxVal - minVal));
                        }
                    }
                    else
                    {
                        child = (double[])p1.Clone();
                    }

                    for (int d = 0; d < dim; d++)
                    {
                        if (GlobalRng.NextDouble() < mutationRate)
                            child[d] = Clamp(child[d] + NextGaussian(0, sigma));
                    }
                    newPopulation.Add(child);
                }
                population = newPopulation;
            }

            return (bestInd, bestScore);
        }

        static string Fmt(double v, string f = "F2") { return v.ToString(f, IC); }

        static double Median(List<double> xs)
        {
            var s = xs.OrderBy(v => v).ToList();
            int n = s.Count;
            return n % 2 == 1 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
        }

        static (double lo, double hi) Wilson(int k, int n)
        {
            if (n == 0) return (0, 0);
            double z = 1.96, p = (double)k / n;
            double den = 1.0 + z * z / n;
            double center = (p + z * z / (2.0 * n)) / den;
            double half = z * Math.Sqrt(p * (1 - p) / n + z * z / (4.0 * n * n)) / den;
            return (Math.Max(0, center - half) * 100.0, Math.Min(1, center + half) * 100.0);
        }

        static void RunPart1(int runs = 20)
        {
            Console.WriteLine(new string('=', 78));
            Console.WriteLine($"ЧАСТИНА 1: розривна функція, dim={DIM}, бюджет {BUDGET} викликів f, {runs} запусків");
            Console.WriteLine(new string('=', 78));

            var results = new Dictionary<string, List<double>>
            {
                { "GD_single", new List<double>() },
                { "GD_multistart", new List<double>() },
                { "GA", new List<double>() }
            };
            var gaHists = new List<List<double>>();
            var gdTraces = new List<List<(int cost, double best)>>();
            var gaEvalsToSuccess = new List<double>();
            const int GA_POP = 50;

            for (int r = 0; r < runs; r++)
            {
                Seed(100 + r);
                var f1 = new CountedFunction(ShiftedDiscontinuousRastrigin);
                results["GD_single"].Add(GradientDescent(f1).value);

                Seed(100 + r);
                var f2 = new CountedFunction(ShiftedDiscontinuousRastrigin);
                var trace = new List<(int cost, double best)>();
                results["GD_multistart"].Add(GradientDescentMultistart(f2, BUDGET, 0.01, trace).value);
                gdTraces.Add(trace);

                Seed(100 + r);
                var f3 = new CountedFunction(ShiftedDiscontinuousRastrigin);
                var hist = new List<double>();
                results["GA"].Add(ContinuousGa(f3, BUDGET, popSize: GA_POP, history: hist).value);
                gaHists.Add(hist);
                int idx = hist.FindIndex(h => h < SUCCESS_EPS);
                if (idx >= 0) gaEvalsToSuccess.Add((idx + 1) * GA_POP);
            }

            Console.WriteLine($"{"Метод",-16}{"Сер.",10}{"Мед.",10}{"Мін.",10}{"Макс.",10}{"Std",9}{"Успіх(f<0.1)",14}");
            using (var writer = new StreamWriter("part1_results.csv"))
            {
                writer.WriteLine("Method,Mean,Median,Min,Max,Std,Success_Rate_Percent");
                foreach (var pair in results)
                {
                    string method = pair.Key;
                    var vals = pair.Value;
                    vals.Sort();
                    double mean = vals.Average();
                    double median = Median(vals);
                    double min = vals.First();
                    double max = vals.Last();
                    double std = Math.Sqrt(vals.Select(v => Math.Pow(v - mean, 2)).Average());
                    double success = (double)vals.Count(v => v < SUCCESS_EPS) / runs * 100.0;

                    Console.WriteLine($"{method,-16}{mean,10:F4}{median,10:F4}{min,10:F4}{max,10:F4}{std,9:F4}{success,13:F0}%");
                    writer.WriteLine(string.Format(IC, "{0},{1:F4},{2:F4},{3:F4},{4:F4},{5:F4},{6:F1}",
                        method, mean, median, min, max, std, success));
                }
            }
            if (gaEvalsToSuccess.Count > 0)
                Console.WriteLine($"ГА: сер. кількість викликів f до f<{SUCCESS_EPS}: {gaEvalsToSuccess.Average():F0} (успішних запусків {gaEvalsToSuccess.Count}/{runs})");

            using (var writer = new StreamWriter("part1_convergence.csv"))
            {
                writer.WriteLine("Evals,GA_Mean,GA_Median,GDmulti_Mean,GDmulti_Median");
                for (int c = 1000; c <= BUDGET; c += 1000)
                {
                    int gi = c / GA_POP - 1;
                    var ga = gaHists.Select(h => h[gi]).ToList();
                    var gd = new List<double>();
                    foreach (var tr in gdTraces)
                    {
                        var done = tr.Where(t => t.cost <= c).ToList();
                        if (done.Count > 0) gd.Add(done.Last().best);
                    }
                    string gdMean = gd.Count == runs ? Fmt(gd.Average(), "F4") : "";
                    string gdMed = gd.Count == runs ? Fmt(Median(gd), "F4") : "";
                    writer.WriteLine(string.Join(",", new[] { c.ToString(), Fmt(ga.Average(), "F4"), Fmt(Median(ga), "F4"), gdMean, gdMed }));
                }
            }
            Console.WriteLine("[OK] part1_results.csv, part1_convergence.csv збережено.");
        }

        static void RunPart1Sensitivity(int runs = 20)
        {
            Console.WriteLine("\n" + new string('=', 78));
            Console.WriteLine($"ЧАСТИНА 1: чутливість ГА та градієнтного спуску ({runs} запусків на значення, бюджет {BUDGET})");
            Console.WriteLine(new string('=', 78));

            var rows = new List<string>();

            void RunGa(string param, double v, int pop, double pc, double pm)
            {
                var vals = new List<double>();
                var evs = new List<double>();
                for (int r = 0; r < runs; r++)
                {
                    Seed(500 + r);
                    var f = new CountedFunction(ShiftedDiscontinuousRastrigin);
                    var hist = new List<double>();
                    var res = ContinuousGa(f, BUDGET, popSize: pop, crossoverRate: pc, mutationRate: pm, history: hist);
                    vals.Add(res.value);
                    int idx = hist.FindIndex(h => h < SUCCESS_EPS);
                    if (idx >= 0) evs.Add((idx + 1) * pop);
                }
                double succ = vals.Count(x => x < SUCCESS_EPS) * 100.0 / runs;
                string evStr = evs.Count > 0 ? Fmt(evs.Average(), "F0") : "N/A";
                Console.WriteLine($"  GA {param}={v,-6}: успіх {succ:F0}% | сер. f {vals.Average():F4} | мед. f {Median(vals):F4} | викликів до успіху {evStr}");
                rows.Add(string.Join(",", new[] { "GA", param, Fmt(v, "G"), Fmt(succ, "F1"), Fmt(vals.Average(), "F4"), Fmt(Median(vals), "F4"), evStr }));
            }

            foreach (var pm in new[] { 0.05, 0.1, 0.2, 0.4, 0.6 }) RunGa("mutation_rate", pm, 50, 0.9, pm);
            foreach (var pop in new[] { 20, 50, 100, 200 }) RunGa("pop_size", pop, pop, 0.9, 0.2);
            foreach (var pc in new[] { 0.3, 0.6, 0.9 }) RunGa("crossover_rate", pc, 50, pc, 0.2);

            foreach (var lr in new[] { 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1 })
            {
                var vals = new List<double>();
                for (int r = 0; r < runs; r++)
                {
                    Seed(700 + r);
                    var f = new CountedFunction(ShiftedDiscontinuousRastrigin);
                    vals.Add(GradientDescentMultistart(f, BUDGET, lr).value);
                }
                double succ = vals.Count(x => x < SUCCESS_EPS) * 100.0 / runs;
                Console.WriteLine($"  GD_multistart lr={lr,-7}: успіх {succ:F0}% | сер. f {vals.Average():F4} | мед. f {Median(vals):F4} | мін. {vals.Min():F4}");
                rows.Add(string.Join(",", new[] { "GD_multistart", "lr", Fmt(lr, "G"), Fmt(succ, "F1"), Fmt(vals.Average(), "F4"), Fmt(Median(vals), "F4"), "N/A" }));
            }

            using (var writer = new StreamWriter("part1_sensitivity.csv"))
            {
                writer.WriteLine("Method,Parameter,Value,Success_Rate_Percent,Mean_f,Median_f,Avg_Evals_To_Success");
                foreach (var r in rows) writer.WriteLine(r);
            }
            Console.WriteLine("[OK] part1_sensitivity.csv збережено.");
        }

        // =====================================================================
        // ЧАСТИНА 2: ДИСКРЕТНА ЗАДАЧА СКЛАДАННЯ РОЗКЛАДУ
        // =====================================================================
        const int HARD_W = 1000;
        const int WINDOW_W = 15;
        const int OVERLOAD_W = 10;
        const int MAX_PER_DAY = 2;

        // Єдиний обчислювальний бюджет (кількість оцінок fitness) для всіх конфігурацій ГА
        // та для багатостартового жадібного алгоритму: Generations = EQUAL_BUDGET / PopSize.
        const int EQUAL_BUDGET = 6000;

        static readonly string[] HARD_KEYS = { "teacher", "group", "room", "forbidden", "room_type" };
        static readonly Dictionary<string, string> HARD_LABELS = new Dictionary<string, string>
        {
            { "teacher", "Конфлікти викладачів" },
            { "group", "Конфлікти груп" },
            { "room", "Конфлікти аудиторій" },
            { "forbidden", "Заборонені слоти викладачів" },
            { "room_type", "Невідповідний тип аудиторії" },
            { "windows", "Вікна у групах" },
            { "overload", "Перевантаження дня" }
        };

        class Lesson
        {
            public int Id { get; set; }
            public string Subject { get; set; }
            public int TeacherId { get; set; }
            public int GroupId { get; set; }
            public bool NeedsLab { get; set; }

            public Lesson(int id, string subject, int teacherId, int groupId, bool needsLab)
            {
                Id = id; Subject = subject; TeacherId = teacherId; GroupId = groupId; NeedsLab = needsLab;
            }
        }

        static readonly string[] TEACHERS = {
            "Проф. Іваненко", "Доц. Петренко", "Асист. Сидоренко", "Доц. Ковальчук",
            "Проф. Гриценко", "Доц. Бондар", "Доц. Мельник", "Проф. Шевчук"
        };

        static readonly (string subject, int teacher, bool isLab)[] SUBJECTS = {
            ("Матаналіз", 0, false), ("Програмування", 1, true), ("Алгебра", 2, false),
            ("Дискретка", 3, false), ("Бази даних", 1, true), ("ШІ", 2, false),
            ("Фізика", 4, true), ("Мережі", 5, true), ("Статистика", 6, false),
            ("Оптимізація", 7, false)
        };

        class Dataset
        {
            public List<string> Groups { get; set; }
            public string[] Teachers { get; set; }
            public List<(string name, bool isLab)> Rooms { get; set; }
            public int Days { get; set; }
            public int SlotsPerDay { get; set; }
            public List<Lesson> Lessons { get; set; }
            public Dictionary<int, HashSet<int>> Forbidden { get; set; }
        }

        static Dataset BuildDataset(int nGroups, int nSubjects, int lessonsPerGroup,
            List<(string name, bool isLab)> rooms, int days, int slotsPerDay, int unavailPerTeacher, int seed)
        {
            Random rng = new Random(seed);
            List<Lesson> lessons = new List<Lesson>();

            for (int g = 0; g < nGroups; g++)
            {
                var chosenSubjects = Enumerable.Range(0, nSubjects).OrderBy(_ => rng.Next()).Take(lessonsPerGroup).ToList();
                foreach (int s in chosenSubjects)
                {
                    var (subject, teacher, lab) = SUBJECTS[s];
                    lessons.Add(new Lesson(lessons.Count, subject, teacher, g, lab));
                }
            }

            int totalSlots = days * slotsPerDay;
            var forbidden = new Dictionary<int, HashSet<int>>();
            for (int t = 0; t < TEACHERS.Length; t++)
            {
                var slots = Enumerable.Range(0, totalSlots).OrderBy(_ => rng.Next()).Take(unavailPerTeacher);
                forbidden[t] = new HashSet<int>(slots);
            }

            return new Dataset
            {
                Groups = Enumerable.Range(1, nGroups).Select(i => $"Група-{i}").ToList(),
                Teachers = TEACHERS,
                Rooms = rooms,
                Days = days,
                SlotsPerDay = slotsPerDay,
                Lessons = lessons,
                Forbidden = forbidden
            };
        }

        static readonly Dictionary<string, Dataset> DATASETS = new Dictionary<string, Dataset>
        {
            { "Small", BuildDataset(2, 6, 4, new List<(string, bool)>{ ("Ауд. 101", false), ("Лаб. 201", true) }, 5, 3, 2, 1) },
            { "Medium", BuildDataset(4, 8, 5, new List<(string, bool)>{ ("Ауд. 101", false), ("Ауд. 102", false), ("Лаб. 201", true) }, 5, 4, 4, 2) },
            { "Dense", BuildDataset(6, 10, 7, new List<(string, bool)>{ ("Ауд. 101", false), ("Ауд. 102", false), ("Лаб. 201", true), ("Лаб. 202", true) }, 5, 4, 5, 3) }
        };

        class EvaluationResult
        {
            public Dictionary<string, int> Details { get; set; } = new Dictionary<string, int>();
            public int Hard { get; set; }
            public int Soft { get; set; }
            public int Fitness { get; set; }
        }

        class Config
        {
            public string Name { get; set; }
            public string Description { get; set; }
            public int PopSize { get; set; }
            public int Generations { get; set; }
            public double CrossoverRate { get; set; }
            public double MutationRate { get; set; }
            public int EliteCount { get; set; }
            public int TournamentSize { get; set; } = 3;

            public double RepairProb { get; set; }       
            public double GreedySeedFrac { get; set; }   
            public int StagnationGens { get; set; }      
            public bool FeasPriority { get; set; }       

            public Config Clone() { return (Config)MemberwiseClone(); }
        }

        static Config Cfg(string name, string desc, int pop, double pc, double pm, int elite,
            double repair = 0.0, double seedFrac = 0.0, int stagnation = 0, bool feasPriority = false)
        {
            return new Config
            {
                Name = name, Description = desc, PopSize = pop, Generations = EQUAL_BUDGET / pop,
                CrossoverRate = pc, MutationRate = pm, EliteCount = elite,
                RepairProb = repair, GreedySeedFrac = seedFrac, StagnationGens = stagnation, FeasPriority = feasPriority
            };
        }

        static readonly List<Config> CONFIGS = new List<Config>
        {
            Cfg("Config_A", "Базова", 40, 0.80, 0.05, 2),
            Cfg("Config_B", "Висока мутація", 60, 0.65, 0.20, 1),
            Cfg("Config_C", "Велика популяція + елітизм", 100, 0.90, 0.08, 5),
            Cfg("Config_D", "Невдала", 20, 0.50, 0.50, 0),
            Cfg("Config_E", "A + пріоритет допустимості + ремонт", 40, 0.80, 0.05, 2, repair: 0.3, feasPriority: true),
            Cfg("Config_F", "E + жадібна ініціалізація + перезапуск", 40, 0.80, 0.05, 2, repair: 0.3, seedFrac: 0.2, stagnation: 25, feasPriority: true)
        };

        class ScheduleProblem
        {
            public Dataset Data { get; }
            public int NumLessons => Data.Lessons.Count;
            public int TotalSlots => Data.Days * Data.SlotsPerDay;
            public int NumRooms => Data.Rooms.Count;

            public ScheduleProblem(string datasetName)
            {
                Data = DATASETS[datasetName];
            }

            public (int slot, int room) RandomGene()
            {
                return (GlobalRng.Next(TotalSlots), GlobalRng.Next(NumRooms));
            }

            public (int slot, int room)[] RandomChromosome()
            {
                var ch = new (int slot, int room)[NumLessons];
                for (int j = 0; j < NumLessons; j++) ch[j] = RandomGene();
                return ch;
            }

            public EvaluationResult Evaluate((int slot, int room)[] chromosome)
            {
                var res = new EvaluationResult();
                foreach (var k in HARD_KEYS) res.Details[k] = 0;
                res.Details["windows"] = 0;
                res.Details["overload"] = 0;

                if (chromosome == null || chromosome.Length != NumLessons)
                {
                    res.Hard = 999999;
                    res.Fitness = 999999;
                    return res;
                }

                var teacherSched = new Dictionary<int, HashSet<int>>();
                var groupSched = new Dictionary<int, HashSet<int>>();
                var roomSched = new Dictionary<int, HashSet<int>>();
                var groupDay = new Dictionary<int, Dictionary<int, List<int>>>();

                for (int i = 0; i < chromosome.Length; i++)
                {
                    var (slot, room) = chromosome[i];
                    var lesson = Data.Lessons[i];

                    if (!teacherSched.ContainsKey(lesson.TeacherId)) teacherSched[lesson.TeacherId] = new HashSet<int>();
                    if (teacherSched[lesson.TeacherId].Contains(slot)) res.Details["teacher"]++;
                    else teacherSched[lesson.TeacherId].Add(slot);

                    if (!groupSched.ContainsKey(lesson.GroupId)) groupSched[lesson.GroupId] = new HashSet<int>();
                    if (groupSched[lesson.GroupId].Contains(slot)) res.Details["group"]++;
                    else groupSched[lesson.GroupId].Add(slot);

                    if (!roomSched.ContainsKey(room)) roomSched[room] = new HashSet<int>();
                    if (roomSched[room].Contains(slot)) res.Details["room"]++;
                    else roomSched[room].Add(slot);

                    if (Data.Forbidden[lesson.TeacherId].Contains(slot)) res.Details["forbidden"]++;

                    if (lesson.NeedsLab && !Data.Rooms[room].isLab) res.Details["room_type"]++;

                    int day = slot / Data.SlotsPerDay;
                    int slotInDay = slot % Data.SlotsPerDay;
                    if (!groupDay.ContainsKey(lesson.GroupId)) groupDay[lesson.GroupId] = new Dictionary<int, List<int>>();
                    if (!groupDay[lesson.GroupId].ContainsKey(day)) groupDay[lesson.GroupId][day] = new List<int>();
                    groupDay[lesson.GroupId][day].Add(slotInDay);
                }

                foreach (var days in groupDay.Values)
                {
                    foreach (var sList in days.Values)
                    {
                        var uniq = sList.Distinct().ToList();
                        if (uniq.Count > 1)
                            res.Details["windows"] += (uniq.Max() - uniq.Min() + 1) - uniq.Count;
                        if (sList.Count > MAX_PER_DAY)
                            res.Details["overload"] += sList.Count - MAX_PER_DAY;
                    }
                }

                res.Hard = HARD_KEYS.Sum(k => res.Details[k]);
                res.Soft = res.Details["windows"] * WINDOW_W + res.Details["overload"] * OVERLOAD_W;
                res.Fitness = res.Hard * HARD_W + res.Soft;
                return res;
            }

            bool Fits(Lesson l, int slot, int room, HashSet<(int, int)> tBusy, HashSet<(int, int)> gBusy, HashSet<(int, int)> rBusy)
            {
                if (Data.Forbidden[l.TeacherId].Contains(slot)) return false;
                if (tBusy.Contains((l.TeacherId, slot))) return false;
                if (gBusy.Contains((l.GroupId, slot))) return false;
                if (rBusy.Contains((room, slot))) return false;
                if (l.NeedsLab && !Data.Rooms[room].isLab) return false;
                return true;
            }

            public void Repair((int slot, int room)[] ch)
            {
                var tBusy = new HashSet<(int, int)>();
                var gBusy = new HashSet<(int, int)>();
                var rBusy = new HashSet<(int, int)>();
                for (int i = 0; i < ch.Length; i++)
                {
                    var l = Data.Lessons[i];
                    int slot = ch[i].slot, room = ch[i].room;
                    if (!Fits(l, slot, room, tBusy, gBusy, rBusy))
                    {
                        bool found = false;
                        for (int ds = 0; ds < TotalSlots && !found; ds++)
                        {
                            int s = (slot + ds) % TotalSlots;
                            for (int dr = 0; dr < NumRooms; dr++)
                            {
                                int r = (room + dr) % NumRooms;
                                if (Fits(l, s, r, tBusy, gBusy, rBusy))
                                {
                                    slot = s; room = r; found = true;
                                    break;
                                }
                            }
                        }
                        ch[i] = (slot, room);
                    }
                    tBusy.Add((l.TeacherId, slot));
                    gBusy.Add((l.GroupId, slot));
                    rBusy.Add((room, slot));
                }
            }

            
            public ((int slot, int room)[] chromosome, EvaluationResult eval) GreedySolve(bool randomOrder = false)
            {
                int n = NumLessons;
                var chromosome = new (int slot, int room)[n];
                var order = Enumerable.Range(0, n).ToList();
                if (randomOrder) order = order.OrderBy(_ => GlobalRng.Next()).ToList();
                int start = randomOrder ? GlobalRng.Next(TotalSlots) : 0;

                var teacherBusy = new Dictionary<int, HashSet<int>>();
                var groupBusy = new Dictionary<int, HashSet<int>>();
                var roomBusy = new Dictionary<int, HashSet<int>>();

                foreach (int idx in order)
                {
                    var lesson = Data.Lessons[idx];
                    bool placed = false;
                    for (int k = 0; k < TotalSlots; k++)
                    {
                        int slot = (start + k) % TotalSlots;
                        if (Data.Forbidden[lesson.TeacherId].Contains(slot)
                            || (teacherBusy.ContainsKey(lesson.TeacherId) && teacherBusy[lesson.TeacherId].Contains(slot))
                            || (groupBusy.ContainsKey(lesson.GroupId) && groupBusy[lesson.GroupId].Contains(slot)))
                            continue;

                        for (int room = 0; room < NumRooms; room++)
                        {
                            if (lesson.NeedsLab && !Data.Rooms[room].isLab) continue;
                            if (roomBusy.ContainsKey(room) && roomBusy[room].Contains(slot)) continue;

                            chromosome[idx] = (slot, room);
                            if (!teacherBusy.ContainsKey(lesson.TeacherId)) teacherBusy[lesson.TeacherId] = new HashSet<int>();
                            teacherBusy[lesson.TeacherId].Add(slot);

                            if (!groupBusy.ContainsKey(lesson.GroupId)) groupBusy[lesson.GroupId] = new HashSet<int>();
                            groupBusy[lesson.GroupId].Add(slot);

                            if (!roomBusy.ContainsKey(room)) roomBusy[room] = new HashSet<int>();
                            roomBusy[room].Add(slot);

                            placed = true;
                            break;
                        }
                        if (placed) break;
                    }
                    if (!placed) chromosome[idx] = RandomGene();
                }

                return (chromosome, Evaluate(chromosome));
            }

            public ((int slot, int room)[] chromosome, EvaluationResult eval, int? evalsFeas, int? evalsZero) GreedyMultistart(int budget)
            {
                (int slot, int room)[] bestCh = null;
                EvaluationResult bestEv = null;
                int? ef = null, ez = null;
                for (int k = 1; k <= budget; k++)
                {
                    var g = GreedySolve(true);
                    if (bestEv == null || g.eval.Fitness < bestEv.Fitness)
                    {
                        bestEv = g.eval;
                        bestCh = g.chromosome;
                    }
                    if (ef == null && bestEv.Hard == 0) ef = k;
                    if (bestEv.Fitness == 0) { ez = k; break; }
                }
                return (bestCh, bestEv, ef, ez);
            }

            public class GaResult
            {
                public (int slot, int room)[] Chromosome { get; set; }
                public EvaluationResult Eval { get; set; }
                public int? GenToFeasible { get; set; }
                public int? GenToZero { get; set; }
                public int? EvalsToFeasible { get; set; }
                public int? EvalsToZero { get; set; }
                public List<int> HistFit { get; set; } = new List<int>();
                public List<double> HistFeas { get; set; } = new List<double>();
            }

            public GaResult GeneticSolve(Config cfg)
            {
                int popSize = cfg.PopSize;
                int generations = cfg.Generations;

                Func<EvaluationResult, long> key = ev => cfg.FeasPriority ? (long)ev.Hard * 1000000L + ev.Soft : (long)ev.Fitness;

                var population = new List<(int slot, int room)[]>();
                int nSeed = (int)Math.Round(cfg.GreedySeedFrac * popSize);
                for (int i = 0; i < popSize; i++)
                    population.Add(i < nSeed ? GreedySolve(true).chromosome : RandomChromosome());

                (int slot, int room)[] bestChrom = null;
                EvaluationResult bestEval = null;
                int? genToFeasible = null, genToZero = null;
                int? evalsToFeasible = null, evalsToZero = null;
                var histFit = new List<int>();
                var histFeas = new List<double>();
                int lastImprove = 0;

                for (int gen = 0; gen < generations; gen++)
                {
                    var evaluated = population.Select(ch => (ch, ev: Evaluate(ch))).OrderBy(x => key(x.ev)).ToList();
                    int evals = popSize * (gen + 1);

                    var top = evaluated[0];
                    if (bestEval == null || key(top.ev) < key(bestEval))
                    {
                        bestChrom = ((int slot, int room)[])top.ch.Clone();
                        bestEval = top.ev;
                        lastImprove = gen;
                    }

                    histFit.Add(bestEval.Fitness);
                    histFeas.Add((double)evaluated.Count(x => x.ev.Hard == 0) / popSize);

                    if (genToFeasible == null && bestEval.Hard == 0)
                    {
                        genToFeasible = gen + 1;
                        evalsToFeasible = evals;
                    }

                    if (bestEval.Fitness == 0)
                    {
                        genToZero = gen + 1;
                        evalsToZero = evals;
                        break;
                    }

                    var newPop = new List<(int slot, int room)[]>();
                    for (int k = 0; k < Math.Min(cfg.EliteCount, popSize); k++)
                        newPop.Add(((int slot, int room)[])evaluated[k].ch.Clone());

                    (int slot, int room)[] TourSelect()
                    {
                        (int slot, int room)[] pBest = null;
                        long minF = long.MaxValue;
                        for (int t = 0; t < cfg.TournamentSize; t++)
                        {
                            var cand = evaluated[GlobalRng.Next(evaluated.Count)];
                            long kf = key(cand.ev);
                            if (kf < minF)
                            {
                                minF = kf;
                                pBest = cand.ch;
                            }
                        }
                        return pBest;
                    }

                    while (newPop.Count < popSize)
                    {
                        var p1 = TourSelect();
                        var p2 = TourSelect();
                        var child = new (int slot, int room)[NumLessons];

                        if (GlobalRng.NextDouble() < cfg.CrossoverRate)
                        {
                            int pt1 = GlobalRng.Next(NumLessons);
                            int pt2 = GlobalRng.Next(pt1, NumLessons + 1);
                            for (int i = 0; i < NumLessons; i++)
                                child[i] = (i >= pt1 && i < pt2) ? p2[i] : p1[i];
                        }
                        else
                        {
                            Array.Copy(p1, child, NumLessons);
                        }

                        for (int i = 0; i < NumLessons; i++)
                        {
                            if (GlobalRng.NextDouble() < cfg.MutationRate)
                            {
                                var (slot, room) = child[i];
                                double mode = GlobalRng.NextDouble();
                                if (mode < 0.4) slot = GlobalRng.Next(TotalSlots);
                                else if (mode < 0.7) room = GlobalRng.Next(NumRooms);
                                else (slot, room) = RandomGene();
                                child[i] = (slot, room);
                            }
                        }

                        if (cfg.RepairProb > 0.0 && GlobalRng.NextDouble() < cfg.RepairProb)
                            Repair(child);

                        newPop.Add(child);
                    }

                    if (cfg.StagnationGens > 0 && gen - lastImprove >= cfg.StagnationGens)
                    {
                        int from = Math.Max(cfg.EliteCount, popSize / 2);
                        for (int k = from; k < popSize; k++)
                            newPop[k] = RandomChromosome();
                        lastImprove = gen;
                    }

                    population = newPop;
                }

                while (histFit.Count < generations)
                {
                    histFit.Add(histFit.Last());
                    histFeas.Add(histFeas.Last());
                }

                return new GaResult
                {
                    Chromosome = bestChrom,
                    Eval = bestEval,
                    GenToFeasible = genToFeasible,
                    GenToZero = genToZero,
                    EvalsToFeasible = evalsToFeasible,
                    EvalsToZero = evalsToZero,
                    HistFit = histFit,
                    HistFeas = histFeas
                };
            }
        }

        class RunStat
        {
            public int Fitness, Hard, Soft;
            public int? GenFeas, EvalsFeas, EvalsZero;
        }

        static readonly string[] SUMMARY_HEADER = {
            "Dataset", "Config", "Runs", "Feasible_Rate_Percent", "Feasible_CI95_Low", "Feasible_CI95_High",
            "Zero_Fitness_Rate_Percent", "Avg_Fitness", "Median_Fitness", "Min_Fitness", "Max_Fitness", "Std_Fitness",
            "Avg_Soft_Feasible", "Avg_Gen_To_Feasible", "Avg_Evals_To_Feasible", "Avg_Evals_To_Zero", "Max_Evals"
        };

        static string[] MakeRow(string ds, string cfg, List<RunStat> rs, int maxEvals)
        {
            int n = rs.Count;
            var fits = rs.Select(r => (double)r.Fitness).ToList();
            var feas = rs.Where(r => r.Hard == 0).ToList();
            double feasRate = 100.0 * feas.Count / n;
            var ci = Wilson(feas.Count, n);
            double zeroRate = 100.0 * rs.Count(r => r.Fitness == 0) / n;
            double avg = fits.Average();
            double std = Math.Sqrt(fits.Select(v => (v - avg) * (v - avg)).Average());
            string softF = feas.Any() ? Fmt(feas.Average(r => (double)r.Soft), "F1") : "N/A";
            string genF = rs.Any(r => r.GenFeas.HasValue) ? Fmt(rs.Where(r => r.GenFeas.HasValue).Average(r => (double)r.GenFeas.Value), "F1") : "N/A";
            string evF = rs.Any(r => r.EvalsFeas.HasValue) ? Fmt(rs.Where(r => r.EvalsFeas.HasValue).Average(r => (double)r.EvalsFeas.Value), "F0") : "N/A";
            string evZ = rs.Any(r => r.EvalsZero.HasValue) ? Fmt(rs.Where(r => r.EvalsZero.HasValue).Average(r => (double)r.EvalsZero.Value), "F0") : "N/A";
            return new[] {
                ds, cfg, n.ToString(), Fmt(feasRate, "F1"), Fmt(ci.lo, "F1"), Fmt(ci.hi, "F1"),
                Fmt(zeroRate, "F1"), Fmt(avg), Fmt(Median(fits)), Fmt(fits.Min()), Fmt(fits.Max()), Fmt(std),
                softF, genF, evF, evZ, maxEvals.ToString()
            };
        }

        static void PrintRow(string[] r)
        {
            Console.WriteLine($"  [{r[1]}] допустимих {r[3]}% (95% ДІ {r[4]}–{r[5]}) | ідеальних {r[6]}% | " +
                              $"fitness сер. {r[7]}, мед. {r[8]}, мін {r[9]}, макс {r[10]}, std {r[11]} | " +
                              $"soft у допустимих {r[12]} | поколінь до допустимого {r[13]} | " +
                              $"оцінок до допустимого {r[14]}, до ідеалу {r[15]}");
        }

        static RunStat ToStat(ScheduleProblem.GaResult r)
        {
            return new RunStat
            {
                Fitness = r.Eval.Fitness, Hard = r.Eval.Hard, Soft = r.Eval.Soft,
                GenFeas = r.GenToFeasible, EvalsFeas = r.EvalsToFeasible, EvalsZero = r.EvalsToZero
            };
        }

        static string BuildBestSchedule(ScheduleProblem prob, string title, (int slot, int room)[] chromosome, EvaluationResult ev, string source)
        {
            var sb = new StringBuilder();
            sb.AppendLine(new string('=', 78));
            sb.AppendLine($"НАЙКРАЩИЙ РОЗКЛАД: {title} | Fitness = {ev.Fitness} | джерело: {source}");
            sb.AppendLine(new string('=', 78));
            sb.AppendLine("Розшифровка штрафів:");
            foreach (var key in HARD_KEYS)
                sb.AppendLine($"  [жорстке] {HARD_LABELS[key],-32}: {ev.Details[key]} (× {HARD_W} = {ev.Details[key] * HARD_W})");
            sb.AppendLine($"  [м'яке]   {HARD_LABELS["windows"],-32}: {ev.Details["windows"]} (× {WINDOW_W} = {ev.Details["windows"] * WINDOW_W})");
            sb.AppendLine($"  [м'яке]   {HARD_LABELS["overload"],-32}: {ev.Details["overload"]} (× {OVERLOAD_W} = {ev.Details["overload"] * OVERLOAD_W})");
            sb.AppendLine($"  Разом: жорстких порушень = {ev.Hard}, м'яких штрафів = {ev.Soft}");
            sb.AppendLine(new string('-', 78));

            string[] daysNames = { "Понеділок", "Вівторок", "Середа", "Четвер", "П'ятниця" };
            int spd = prob.Data.SlotsPerDay;
            var order = Enumerable.Range(0, prob.NumLessons).OrderBy(i => chromosome[i].slot).ThenBy(i => chromosome[i].room).ToList();

            foreach (var idx in order)
            {
                var (slot, room) = chromosome[idx];
                var lesson = prob.Data.Lessons[idx];
                sb.AppendLine($"[{daysNames[slot / spd],-9}, Пара {slot % spd + 1}] | " +
                              $"{prob.Data.Groups[lesson.GroupId],-8} | {lesson.Subject,-14} | " +
                              $"{prob.Data.Teachers[lesson.TeacherId],-17} | {prob.Data.Rooms[room].name}");
            }
            sb.AppendLine();
            return sb.ToString();
        }

        static void RunPart2(int runsPerConfig = 20, int sensitivityRuns = 30)
        {
            Console.WriteLine("\n" + new string('=', 78));
            Console.WriteLine($"ЧАСТИНА 2: розклад ({runsPerConfig} запусків на конфігурацію, бюджет {EQUAL_BUDGET} оцінок fitness)");
            Console.WriteLine("Успіх = допустимий розклад (hard=0). Ідеал = fitness 0 (hard=0 і soft=0).");
            Console.WriteLine(new string('=', 78));

            var summaryRows = new List<string[]>();
            var convLines = new List<string>();
            var bestByDataset = new Dictionary<string, (EvaluationResult ev, (int slot, int room)[] ch, string source)>();

            foreach (var dName in new[] { "Small", "Medium", "Dense" })
            {
                var prob = new ScheduleProblem(dName);
                int nLabs = prob.Data.Lessons.Count(l => l.NeedsLab);
                Console.WriteLine($"\n>>> Датасет: {dName} (занять: {prob.NumLessons}, з них лабораторних: {nLabs}, " +
                                  $"аудиторій: {prob.NumRooms}, слотів: {prob.TotalSlots}, груп: {prob.Data.Groups.Count})");

                Seed(1);
                var g = prob.GreedySolve(false);
                var gStat = new RunStat
                {
                    Fitness = g.eval.Fitness, Hard = g.eval.Hard, Soft = g.eval.Soft,
                    EvalsFeas = g.eval.Hard == 0 ? (int?)1 : null,
                    EvalsZero = g.eval.Fitness == 0 ? (int?)1 : null
                };
                var rowG = MakeRow(dName, "Greedy", new List<RunStat> { gStat }, 1);
                summaryRows.Add(rowG);
                PrintRow(rowG);
                bestByDataset[dName] = (g.eval, g.chromosome, "Greedy");

                var gmStats = new List<RunStat>();
                for (int r = 0; r < runsPerConfig; r++)
                {
                    Seed(1000 + r);
                    var gm = prob.GreedyMultistart(EQUAL_BUDGET);
                    gmStats.Add(new RunStat
                    {
                        Fitness = gm.eval.Fitness, Hard = gm.eval.Hard, Soft = gm.eval.Soft,
                        EvalsFeas = gm.evalsFeas, EvalsZero = gm.evalsZero
                    });
                    if (gm.eval.Fitness < bestByDataset[dName].ev.Fitness)
                        bestByDataset[dName] = (gm.eval, gm.chromosome, "Greedy_multistart");
                }
                var rowGm = MakeRow(dName, "Greedy_multistart", gmStats, EQUAL_BUDGET);
                summaryRows.Add(rowGm);
                PrintRow(rowGm);

                foreach (var cfg in CONFIGS)
                {
                    var results = new List<ScheduleProblem.GaResult>();
                    for (int r = 0; r < runsPerConfig; r++)
                    {
                        Seed(1000 + r);
                        results.Add(prob.GeneticSolve(cfg));
                    }

                    var row = MakeRow(dName, cfg.Name, results.Select(x => ToStat(x)).ToList(), cfg.PopSize * cfg.Generations);
                    summaryRows.Add(row);
                    PrintRow(row);

                    foreach (var res in results)
                        if (res.Eval.Fitness < bestByDataset[dName].ev.Fitness)
                            bestByDataset[dName] = (res.Eval, res.Chromosome, "ГА " + cfg.Name);

                    for (int i = 0; i < cfg.Generations; i++)
                    {
                        var fitsAtGen = results.Select(r => (double)r.HistFit[i]).ToList();
                        double avgFeas = results.Average(r => r.HistFeas[i]);
                        convLines.Add(string.Join(",", new[] {
                            dName, cfg.Name, (i + 1).ToString(), (cfg.PopSize * (i + 1)).ToString(),
                            Fmt(fitsAtGen.Average()), Fmt(Median(fitsAtGen)), Fmt(avgFeas, "F4") }));
                    }
                }
            }

            using (var writer = new StreamWriter("ga_results.csv"))
            {
                writer.WriteLine(string.Join(",", SUMMARY_HEADER));
                foreach (var row in summaryRows) writer.WriteLine(string.Join(",", row));
            }
            using (var writer = new StreamWriter("convergence_long.csv"))
            {
                writer.WriteLine("Dataset,Config,Generation,Evals,AvgBestFitness,MedianBestFitness,AvgFeasibleFraction");
                foreach (var l in convLines) writer.WriteLine(l);
            }

            Console.WriteLine("\n" + new string('=', 78));
            Console.WriteLine($"АНАЛІЗ ЧУТЛИВОСТІ (Dense, база = Config_A, {sensitivityRuns} запусків на значення, рівний бюджет {EQUAL_BUDGET} оцінок)");
            Console.WriteLine(new string('=', 78));
            var probDense = new ScheduleProblem("Dense");
            var baseCfg = CONFIGS[0];
            var sensRows = new List<string[]>();

            void RunSweep(string paramName, double[] vals, Action<Config, double> setter)
            {
                foreach (var v in vals)
                {
                    var c = baseCfg.Clone();
                    setter(c, v);
                    c.Generations = EQUAL_BUDGET / c.PopSize;
                    var stats = new List<RunStat>();
                    for (int s = 0; s < sensitivityRuns; s++)
                    {
                        Seed(2000 + s);
                        stats.Add(ToStat(probDense.GeneticSolve(c)));
                    }
                    var row = MakeRow("Dense", paramName + "=" + Fmt(v, "G"), stats, c.PopSize * c.Generations);
                    Console.Write($"  {paramName}={v,-6}:");
                    PrintRow(row);
                    sensRows.Add(new[] { paramName, Fmt(v, "G") }.Concat(row.Skip(2)).ToArray());
                }
            }

            RunSweep("mutation_rate", new[] { 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.20 }, (c, v) => c.MutationRate = v);
            RunSweep("pop_size", new[] { 20.0, 40.0, 80.0, 160.0 }, (c, v) => c.PopSize = (int)v);
            RunSweep("crossover_rate", new[] { 0.3, 0.6, 0.9 }, (c, v) => c.CrossoverRate = v);
            RunSweep("elite_count", new[] { 0.0, 1.0, 2.0, 5.0, 10.0 }, (c, v) => c.EliteCount = (int)v);
            RunSweep("repair_prob", new[] { 0.0, 0.1, 0.3, 0.6, 1.0 }, (c, v) => { c.RepairProb = v; c.FeasPriority = true; });

            using (var writer = new StreamWriter("sensitivity.csv"))
            {
                writer.WriteLine("Parameter,Value," + string.Join(",", SUMMARY_HEADER.Skip(2)));
                foreach (var r in sensRows) writer.WriteLine(string.Join(",", r));
            }

            Console.WriteLine("\n[OK] ga_results.csv, convergence_long.csv, sensitivity.csv згенеровано.");

            var all = new StringBuilder();
            foreach (var dName in new[] { "Small", "Medium", "Dense" })
            {
                var best = bestByDataset[dName];
                all.Append(BuildBestSchedule(new ScheduleProblem(dName), dName, best.ch, best.ev, best.source));
            }
            Console.WriteLine("\n" + all);
            File.WriteAllText("best_schedules.txt", all.ToString());
            Console.WriteLine("[OK] best_schedules.txt збережено (повний вивід, без обрізання).");
        }

        static void Main(string[] args)
        {
            CultureInfo.CurrentCulture = IC;
            CultureInfo.DefaultThreadCurrentCulture = IC;
            Console.OutputEncoding = Encoding.UTF8;

            RunPart1(20);
            RunPart1Sensitivity(20);
            RunPart2(20, 30);

#if SCOTTPLOT
            Plots.Make();
#endif
        }
    }
}