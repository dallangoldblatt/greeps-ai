import greenfoot.*;  // (World, Actor, GreenfootImage, and Greenfoot)

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

/**
 * A Greep that communicates with paint.
 *
 * Approach "paint_comm" (policies/paint_comm/; this file is copied to
 * src/Greep.java by `python -m tools.use_policy paint_comm`).
 *
 * The same network as mlp_ppo (policies/mlp_ppo/model.py), trained by
 * policies/paint_comm/train.py with incentives for using paint to talk:
 * greeps were rewarded when paint they spat changed where other greeps went,
 * when their own movement depended on others' paint, and when their choice of
 * colour carried information, with a small cost per drop. The incentives were
 * faded out during training, so what remains is paint use that helps the score.
 *
 * Every act round the greep observes (including the paint colours under it),
 * runs one forward pass, and samples all of its free decisions (heading,
 * memory, flags, paint) and whether to make the one real action, move().
 * Dropping a tomato at the ship and loading other greeps are reflexes.
 *
 * The weights are embedded below (policies/paint_comm/export.py). A
 * policy.weights file in the working directory (or -Dgreeps.policy=<file>)
 * overrides them; the headless trainer swaps weights via setPolicyWeights().
 */
public class Greep extends Creature {

    /** Which approach this is (policies/<name>/). */
    public static final String POLICY_NAME = "paint_comm";

    // ---- Observation layout (same as greeps/torch_sim.py) ----
    private static final int IN_CARRYING = 0;
    private static final int IN_AT_SHIP = 1;
    private static final int IN_AT_FOOD = 2;
    private static final int IN_AT_WATER = 3;
    private static final int IN_AT_EDGE = 4;
    private static final int IN_PAINT = 5;       // 3: red, orange, purple
    private static final int IN_MEMORY = 8;      // 8 bits
    private static final int IN_FLAGS = 16;      // 2
    private static final int IN_HEADING = 18;    // cos, sin of rotation
    private static final int IN_HOME = 20;       // cos, sin of (home bearing - rotation)
    public static final int NUM_INPUTS = 22;

    private static final int HIDDEN = 64;

    // ---- Output layout (logits) ----
    private static final int OUT_MODE = 0;       // 3: keep heading, face home, face pile
    private static final int OUT_TURN = 3;       // 37: turn 10 * (i - 18) degrees
    private static final int OUT_MOVE = 40;      // 1: Bernoulli
    private static final int OUT_MEMORY = 41;    // 8: Bernoulli per bit
    private static final int OUT_FLAGS = 49;     // 2: Bernoulli per flag
    private static final int OUT_SPIT = 51;      // 4: none, red, orange, purple
    public static final int NUM_OUTPUTS = 55;
    private static final int NUM_TURNS = 37;

    /** W1, b1, W2, b2, W3, b3, each weight matrix row-major (output, input). */
    public static final int NUM_PARAMS =
        (NUM_INPUTS + 1) * HIDDEN + (HIDDEN + 1) * HIDDEN + (HIDDEN + 1) * NUM_OUTPUTS;

    private static final String[] PAINT_COLORS = { "red", "orange", "purple" };
    private static final String POLICY_FILE = "policy.weights";
    /** Experiments only (-Dgreeps.paintBlind=true): ignore paint, to measure whether it carries information. */
    private static final boolean PAINT_BLIND = Boolean.getBoolean("greeps.paintBlind");
    /**
     * Diagnostics only (-Dgreeps.obsTrace=<file>): one line per act with the
     * observation, the sampled decisions and the paint drops nearby (for
     * comparing with the GPU simulator). Draws no random
     * numbers, so play is unchanged.
     */
    private static final String OBS_TRACE = System.getProperty("greeps.obsTrace");
    private static java.io.PrintWriter obsTrace;
    private int acts = 0;

    // Loaded on first use: the embedded weights are defined at the end of the
    // file, after this field's initializer would run.
    private static double[] weights;

    /**
     * Default constructor for testing purposes.
     */
    public Greep() {
        this(null);
    }

    /**
     * Create a Greep with its home space ship.
     */
    public Greep(Ship ship) {
        super(ship);
    }

    /**
     * Perform one action.
     */
    @Override
    public void act() {
        super.act();

        // Reflex: a tomato at the ship always counts.
        if (carryingTomato() && atShip()) {
            dropTomato();
        }

        double[] in = observe();
        double[] out = forward(in);
        StringBuilder trace = OBS_TRACE == null ? null : traceStart(in);

        int mode = sampleCategorical(out, OUT_MODE, 3);
        switch (mode) {
            case 1:
                turnHome();
                break;
            case 2:
                turnToTomatoes();
                break;
            default:
                break;
        }
        int turnBin = sampleCategorical(out, OUT_TURN, NUM_TURNS);
        turn(10 * (turnBin - NUM_TURNS / 2));

        int memory = 0;
        for (int bit = 0; bit < 8; bit++) {
            if (sampleBernoulli(out[OUT_MEMORY + bit])) {
                memory |= 1 << bit;
            }
        }
        setMemory(memory);
        setFlag(1, sampleBernoulli(out[OUT_FLAGS]));
        setFlag(2, sampleBernoulli(out[OUT_FLAGS + 1]));

        int spit = sampleCategorical(out, OUT_SPIT, 4);
        if (spit > 0) {
            spit(PAINT_COLORS[spit - 1]);
        }

        boolean moving = sampleBernoulli(out[OUT_MOVE]);
        if (moving) {
            move();
        }
        if (trace != null) {
            trace.append(" A ").append(mode).append(' ').append(turnBin).append(' ')
                .append(memory).append(' ').append(getFlag(1) ? 1 : 0).append(getFlag(2) ? 1 : 0)
                .append(' ').append(spit).append(' ').append(moving ? 1 : 0)
                .append(" T ").append(getX()).append(' ').append(getY()).append(' ')
                .append(getRotation()).append(' ').append(atWater() ? 1 : 0);
            writeTrace(trace);
        }

        // Reflex: hand a tomato to another greep if we share a pile.
        loadTomato();
    }

    /**
     * Trace line start: map (ship position), this greep's act count, state,
     * observation, and every paint drop within 40 px (x, y, colour index,
     * whether Greenfoot counts it as touching).
     */
    private StringBuilder traceStart(double[] in) {
        Actor ship = getWorld().getObjects(Ship.class).get(0);
        StringBuilder line = new StringBuilder("O ")
            .append(ship.getX()).append(' ').append(ship.getY()).append(' ')
            .append(acts++).append(' ')
            .append(getX()).append(' ').append(getY()).append(' ').append(getRotation())
            .append(" I");
        for (int i = 0; i < NUM_INPUTS; i++) {
            line.append(' ').append(i < IN_HEADING ? String.valueOf((int) in[i])
                                                   : String.format(java.util.Locale.ROOT, "%.4f", in[i]));
        }
        List<Paint> touching = getIntersectingObjects(Paint.class);
        line.append(" P");
        for (Paint paint : getObjectsInRange(40, Paint.class)) {
            int colour = java.util.Arrays.asList(PAINT_COLORS).indexOf(paint.getColor());
            line.append(' ').append(paint.getX()).append(',').append(paint.getY()).append(',')
                .append(colour).append(',').append(touching.contains(paint) ? 1 : 0);
        }
        return line;
    }

    private static synchronized void writeTrace(StringBuilder line) {
        if (obsTrace == null) {
            try {
                obsTrace = new java.io.PrintWriter(new java.io.FileWriter(OBS_TRACE, true));
            } catch (IOException e) {
                throw new IllegalStateException("Cannot open " + OBS_TRACE, e);
            }
            Runtime.getRuntime().addShutdownHook(new Thread(obsTrace::flush));
        }
        obsTrace.println(line);
    }

    private double[] observe() {
        double[] in = new double[NUM_INPUTS];

        in[IN_CARRYING] = carryingTomato() ? 1 : 0;
        in[IN_AT_SHIP] = atShip() ? 1 : 0;
        in[IN_AT_FOOD] = seeFood() ? 1 : 0;
        in[IN_AT_WATER] = atWater() ? 1 : 0;
        in[IN_AT_EDGE] = atWorldEdge() ? 1 : 0;
        // Same as seePaint(color) for each color, with one collision query.
        for (Paint paint : PAINT_BLIND ? List.<Paint>of() : getIntersectingObjects(Paint.class)) {
            for (int i = 0; i < PAINT_COLORS.length; i++) {
                if (paint.getColor().equals(PAINT_COLORS[i])) {
                    in[IN_PAINT + i] = 1;
                }
            }
        }

        int memory = getMemory();
        for (int bit = 0; bit < 8; bit++) {
            in[IN_MEMORY + bit] = (memory >> bit) & 1;
        }
        in[IN_FLAGS] = getFlag(1) ? 1 : 0;
        in[IN_FLAGS + 1] = getFlag(2) ? 1 : 0;

        int rotation = getRotation();
        double heading = Math.toRadians(rotation);
        in[IN_HEADING] = Math.cos(heading);
        in[IN_HEADING + 1] = Math.sin(heading);

        // Bearing to the ship relative to our heading (via turnHome, since
        // the ship itself is private to Creature).
        turnHome();
        double home = Math.toRadians(getRotation() - rotation);
        setRotation(rotation);
        in[IN_HOME] = Math.cos(home);
        in[IN_HOME + 1] = Math.sin(home);
        return in;
    }

    private static double[] forward(double[] in) {
        if (weights == null) {
            weights = loadWeights();
        }
        int w = 0;
        double[] h1 = new double[HIDDEN];
        w = dense(in, h1, w, true);
        double[] h2 = new double[HIDDEN];
        w = dense(h1, h2, w, true);
        double[] out = new double[NUM_OUTPUTS];
        dense(h2, out, w, false);
        return out;
    }

    /** out = W x + b (optionally tanh) with W, b read from weights[w...]. */
    private static int dense(double[] x, double[] out, int w, boolean tanh) {
        int bias = w + out.length * x.length;
        for (int o = 0; o < out.length; o++) {
            double sum = weights[bias + o];
            for (int i = 0; i < x.length; i++) {
                sum += weights[w++] * x[i];
            }
            out[o] = tanh ? Math.tanh(sum) : sum;
        }
        return bias + out.length;
    }

    /** Uniform random number in [0, 1), from Greenfoot's (seedable) generator. */
    private static double uniform() {
        return Greenfoot.getRandomNumber(1 << 24) / (double) (1 << 24);
    }

    /** Sample from softmax(out[start..start+n)) by inverting its CDF. */
    private static int sampleCategorical(double[] out, int start, int n) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            max = Math.max(max, out[start + i]);
        }
        double[] p = new double[n];
        double total = 0;
        for (int i = 0; i < n; i++) {
            p[i] = Math.exp(out[start + i] - max);
            total += p[i];
        }
        double u = uniform() * total;
        double cumulative = 0;
        for (int i = 0; i < n; i++) {
            cumulative += p[i];
            if (u <= cumulative) {
                return i;
            }
        }
        return n - 1;
    }

    private static boolean sampleBernoulli(double logit) {
        return uniform() < 1 / (1 + Math.exp(-logit));
    }

    /**
     * Check whether a tomato pile intersects this Greep.
     */
    public boolean seeFood() {
        return getOneIntersectingObject(TomatoPile.class) != null;
    }

    /**
     * Face the tomato pile at the current location, if present.
     */
    public void turnToTomatoes() {
        TomatoPile tomatoes =
            (TomatoPile) getOneIntersectingObject(TomatoPile.class);

        if (tomatoes != null) {
            int deltaX = tomatoes.getX() - getX();
            int deltaY = tomatoes.getY() - getY();

            setRotation((int) (
                180 * Math.atan2(deltaY, deltaX) / Math.PI
            ));
        }
    }

    // ---- Policy weights ----

    /** Replace the policy for all greeps (used between training episodes). */
    public static void setPolicyWeights(double[] newWeights) {
        if (newWeights.length != NUM_PARAMS) {
            throw new IllegalArgumentException(
                "Expected " + NUM_PARAMS + " weights, got " + newWeights.length);
        }
        weights = newWeights.clone();
    }

    /** The embedded (trained) policy. */
    public static double[] defaultPolicyWeights() {
        StringBuilder encoded = new StringBuilder();
        for (String chunk : EMBEDDED_WEIGHTS) {
            encoded.append(chunk);
        }
        FloatBuffer floats = ByteBuffer.wrap(Base64.getDecoder().decode(encoded.toString()))
            .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        if (floats.remaining() != NUM_PARAMS) {
            throw new IllegalStateException("Embedded policy has " + floats.remaining()
                + " weights, expected " + NUM_PARAMS);
        }
        double[] w = new double[NUM_PARAMS];
        for (int i = 0; i < w.length; i++) {
            w[i] = floats.get(i);
        }
        return w;
    }

    /** Starting point for black-box fine-tuning: the trained policy itself. */
    public static double[] initialPolicyWeights() {
        return defaultPolicyWeights();
    }

    /** Relative per-weight scale for black-box fine-tuning. */
    public static double[] policySearchScales() {
        double[] scales = new double[NUM_PARAMS];
        java.util.Arrays.fill(scales, 0.1);
        return scales;
    }

    private static double[] loadWeights() {
        // Tests point this at fixture policies with -Dgreeps.policy=<file>.
        String file = System.getProperty("greeps.policy", POLICY_FILE);
        Path path = Paths.get(file);
        if (!Files.exists(path)) {
            if (!file.equals(POLICY_FILE)) {
                throw new IllegalStateException("Policy file not found: " + file);
            }
            return defaultPolicyWeights();
        }
        try {
            List<Double> values = new ArrayList<>();
            for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                for (String token : line.split("\\s+")) {
                    values.add(Double.parseDouble(token));
                }
            }
            double[] loaded = new double[values.size()];
            for (int i = 0; i < loaded.length; i++) {
                loaded[i] = values.get(i);
            }
            if (loaded.length != NUM_PARAMS) {
                throw new IllegalStateException(file + " has " + loaded.length
                    + " weights, expected " + NUM_PARAMS);
            }
            System.out.println("Loaded " + file);
            return loaded;
        } catch (IOException | NumberFormatException e) {
            throw new IllegalStateException("Cannot read " + file, e);
        }
    }

    public static String getAuthorName() {
        return "Dallan Goldblatt";
    }

    @Override
    public String getCurrentImage() {
        if (carryingTomato()) {
            return "greep-with-food.png";
        } else {
            return "greep.png";
        }
    }

    // BEGIN GENERATED WEIGHTS (greeps/policies.py)
    // checkpoints/memory_st_latest.pt: update 1217
    private static final String[] EMBEDDED_WEIGHTS = {
        "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",
    };
    // END GENERATED WEIGHTS
}
