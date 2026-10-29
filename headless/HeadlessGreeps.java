import bluej.Config;
import greenfoot.Actor;
import greenfoot.World;
import greenfoot.WorldVisitor;
import greenfoot.core.ExportedProjectProperties;
import greenfoot.core.Simulation;
import greenfoot.platforms.standalone.ActorDelegateStandAlone;
import greenfoot.platforms.standalone.GreenfootUtilDelegateStandAlone;
import greenfoot.util.GreenfootUtil;
import greenfoot.util.StandalonePropStringManager;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Runs the Greeps scenario without the Greenfoot GUI.
 *
 * The scenario classes and Greenfoot's own World/Actor/collision code are used
 * unchanged; only the simulation thread is replaced by a plain loop that
 * mirrors greenfoot.core.Simulation.runOneLoop. With the same seed this
 * produces exactly the same run as the GUI (see tests/test_headless_parity.py).
 *
 * Usage: HeadlessGreeps [--seed N] [--episodes K] [--trace-dir DIR]
 *
 * Prints one JSON line per episode: {"seed": N, "scores": [m1, m2, m3]}.
 * Episode i uses seed N+i. Without --seed the runs are unseeded. The policy
 * is whatever Greep loads (policy.weights, or its default).
 *
 * Usage: HeadlessGreeps --server
 *
 * Reads one command per line on stdin and answers with one line on stdout
 * (anything the scenario prints goes to stderr instead):
 *
 *   INFO                        -> INFO <num_params> <policy name>
 *   DEFAULT                     -> WEIGHTS <w1> ... <wn> (the weights the Greep plays)
 *   INITIAL                     -> WEIGHTS <w1> ... <wn> (starting point for training)
 *   SCALES                      -> SCALES <s1> ... <sn> (per-weight search scale)
 *   EVAL <seed> <k> <w1> .. <wn> -> SCORES <m1,m2,m3> ... (k episodes, seeds seed..seed+k-1)
 *   QUIT
 *
 * Errors are answered with ERROR <message>.
 *
 * Usage: HeadlessGreeps --dump-water DIR
 *
 * Writes DIR/water<m>.bin for every map: WIDTH*HEIGHT bytes in row-major
 * order, 1 where Earth.isWater(x, y). The GPU simulator (greeps/torch_sim.py)
 * uses these so that its water matches Greenfoot's image decoding exactly.
 */
public final class HeadlessGreeps
{
    // 3 maps x 120 s x 10 acts/s, plus generous slack for ship landing.
    private static final int MAX_ROUNDS = 20000;

    private HeadlessGreeps()
    {
    }

    public static void main(String[] args) throws Exception
    {
        Long seed = null;
        int episodes = 1;
        String traceDir = null;
        boolean server = false;
        String waterDir = null;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--server":
                    server = true;
                    break;
                case "--dump-water":
                    waterDir = args[++i];
                    break;
                case "--seed":
                    seed = Long.parseLong(args[++i]);
                    break;
                case "--episodes":
                    episodes = Integer.parseInt(args[++i]);
                    break;
                case "--trace-dir":
                    traceDir = args[++i];
                    break;
                default:
                    throw new IllegalArgumentException("Unknown argument: " + args[i]);
            }
        }

        // Note: java.awt.headless must stay false. Greenfoot loads images as
        // screen-compatible images (no window is ever opened), and keeping
        // that path identical keeps water detection identical to the GUI.
        System.setProperty("greeps.headless", "true");

        if (server) {
            PrintStream protocol = System.out;
            System.setOut(System.err);
            initialiseGreenfoot();
            serve(protocol);
            return;
        }

        initialiseGreenfoot();
        if (waterDir != null) {
            dumpWater(waterDir);
            return;
        }
        if (traceDir != null) {
            new java.io.File(traceDir).mkdirs();
        }

        for (int episode = 0; episode < episodes; episode++) {
            Long episodeSeed = seed == null ? null : seed + episode;

            if (episodeSeed != null) {
                System.setProperty("greeps.seed", episodeSeed.toString());
            }
            if (traceDir != null) {
                String name = episodeSeed == null
                    ? "trace-" + episode + ".txt"
                    : "trace-seed" + episodeSeed + ".txt";
                System.setProperty("greeps.trace",
                    new java.io.File(traceDir, name).getPath());
            }

            int[] scores = runEpisode();

            StringBuilder json = new StringBuilder("{\"seed\": ");
            json.append(episodeSeed == null ? "null" : episodeSeed.toString());
            json.append(", \"scores\": [");
            for (int i = 0; i < scores.length; i++) {
                json.append(i == 0 ? "" : ", ").append(scores[i]);
            }
            json.append("]}");
            System.out.println(json);
        }
        System.out.flush();
    }

    private static void serve(PrintStream protocol) throws IOException
    {
        BufferedReader in = new BufferedReader(
            new InputStreamReader(System.in, StandardCharsets.UTF_8));
        String line;

        while ((line = in.readLine()) != null) {
            String[] tokens = line.trim().split("\\s+");
            try {
                switch (tokens[0]) {
                    case "INFO":
                        protocol.println("INFO " + Greep.NUM_PARAMS + " " + Greep.POLICY_NAME);
                        break;
                    case "DEFAULT":
                        protocol.println(weightsLine(Greep.defaultPolicyWeights()));
                        break;
                    case "INITIAL":
                        protocol.println(weightsLine(Greep.initialPolicyWeights()));
                        break;
                    case "SCALES":
                        StringBuilder scales = new StringBuilder("SCALES");
                        for (double s : Greep.policySearchScales()) {
                            scales.append(' ').append(s);
                        }
                        protocol.println(scales);
                        break;
                    case "EVAL":
                        protocol.println(evaluate(tokens));
                        break;
                    case "QUIT":
                        return;
                    default:
                        protocol.println("ERROR unknown command " + tokens[0]);
                }
            } catch (RuntimeException e) {
                protocol.println("ERROR " + e.toString().replace('\n', ' '));
            }
            protocol.flush();
        }
    }

    private static String weightsLine(double[] values)
    {
        StringBuilder line = new StringBuilder("WEIGHTS");
        for (double w : values) {
            line.append(' ').append(w);
        }
        return line.toString();
    }

    private static String evaluate(String[] tokens)
    {
        long seed = Long.parseLong(tokens[1]);
        int episodes = Integer.parseInt(tokens[2]);
        double[] weights = new double[tokens.length - 3];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Double.parseDouble(tokens[i + 3]);
        }
        Greep.setPolicyWeights(weights);

        StringBuilder reply = new StringBuilder("SCORES");
        for (int episode = 0; episode < episodes; episode++) {
            System.setProperty("greeps.seed", Long.toString(seed + episode));
            int[] scores = runEpisode();
            reply.append(' ');
            for (int i = 0; i < scores.length; i++) {
                reply.append(i == 0 ? "" : ",").append(scores[i]);
            }
        }
        return reply.toString();
    }

    private static void dumpWater(String dir) throws IOException
    {
        new java.io.File(dir).mkdirs();
        Earth world = new Earth();
        for (int map = 1; map <= world.getScores().length; map++) {
            world.jumpToMap(map);
            byte[] water = new byte[Earth.WIDTH * Earth.HEIGHT];
            for (int y = 0; y < Earth.HEIGHT; y++) {
                for (int x = 0; x < Earth.WIDTH; x++) {
                    water[y * Earth.WIDTH + x] = (byte) (world.isWater(x, y) ? 1 : 0);
                }
            }
            java.nio.file.Files.write(
                java.nio.file.Paths.get(dir, "water" + map + ".bin"), water);
        }
    }

    /**
     * Play all maps of one Earth world and return the per-map scores.
     */
    public static int[] runEpisode()
    {
        Earth world = new Earth();
        int rounds = 0;

        while (!world.isFinished()) {
            if (++rounds > MAX_ROUNDS) {
                throw new IllegalStateException(
                    "Episode did not finish within " + MAX_ROUNDS + " rounds");
            }
            runOneLoop(world);
        }
        return world.getScores();
    }

    /**
     * One act round, as in greenfoot.core.Simulation.runOneLoop: the world
     * acts first, then every actor in act order. The actor list is a snapshot
     * taken after the world acts; actors removed during the round are skipped,
     * and actors added during the round first act in the next round.
     */
    private static void runOneLoop(World world)
    {
        world.act();

        List<Actor> actors =
            new ArrayList<>(WorldVisitor.getObjectsListInActOrder(world));
        for (Actor actor : actors) {
            if (actor.getWorld() != null) {
                actor.act();
            }
        }
    }

    /**
     * The non-GUI part of GreenfootScenarioViewer's start-up: configuration,
     * image loading delegates and the (never started) simulation instance that
     * Greenfoot.delay/stop/setSpeed talk to.
     */
    private static void initialiseGreenfoot() throws IOException
    {
        Properties properties = new Properties();
        try (InputStream in = HeadlessGreeps.class.getClassLoader()
                .getResourceAsStream("standalone.properties")) {
            if (in != null) {
                properties.load(in);
            }
        }
        Config.initializeStandalone(new StandalonePropStringManager(properties));

        GreenfootUtil.initialise(new GreenfootUtilDelegateStandAlone());
        ActorDelegateStandAlone.setupAsActorDelegate();
        ActorDelegateStandAlone.initProperties(new ExportedProjectProperties());

        // The simulation thread is only started by attachWorldHandler, which
        // we never call; while it is not running and not enabled,
        // Greenfoot.delay() returns immediately.
        Simulation.initialize();
    }
}
