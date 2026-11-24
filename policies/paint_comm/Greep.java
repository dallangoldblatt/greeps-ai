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
    // colour_stage2/best.pt: update 1475, GPU-sim total 2777.1 (scrambled paint 729.2)
    private static final String[] EMBEDDED_WEIGHTS = {
        "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",
    };
    // END GENERATED WEIGHTS
}
