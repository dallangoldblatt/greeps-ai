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
 * A Greep controlled by a small neural network.
 *
 * Approach "mlp_ppo" (policies/mlp_ppo/; this file is copied to
 * src/Greep.java by `python -m tools.use_policy mlp_ppo`).
 *
 * The network (policies/mlp_ppo/model.py) was trained with MAPPO on a GPU
 * port of this scenario (policies/mlp_ppo/train.py). During training a critic
 * could see the whole map; the network itself only ever sees what one greep
 * can legally observe: its own sensors, memory and flags.
 *
 * Every act round the greep observes, runs one forward pass, and samples all
 * of its free decisions (heading, memory, flags, paint) and whether to make
 * the one real action, move(). Dropping a tomato at the ship and loading
 * tomatoes onto other greeps are reflexes, since both are free and never hurt.
 *
 * The weights are embedded below (policies/mlp_ppo/export.py). A
 * policy.weights file in the working directory (or -Dgreeps.policy=<file>)
 * overrides them; the headless trainer swaps weights via setPolicyWeights().
 */
public class Greep extends Creature {

    /** Which approach this is (policies/<name>/). */
    public static final String POLICY_NAME = "mlp_ppo";

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

        double[] out = forward(observe());

        switch (sampleCategorical(out, OUT_MODE, 3)) {
            case 1:
                turnHome();
                break;
            case 2:
                turnToTomatoes();
                break;
            default:
                break;
        }
        turn(10 * (sampleCategorical(out, OUT_TURN, NUM_TURNS) - NUM_TURNS / 2));

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

        if (sampleBernoulli(out[OUT_MOVE])) {
            move();
        }

        // Reflex: hand a tomato to another greep if we share a pile.
        loadTomato();
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
    // run1/best.pt: update 950, GPU-sim eval 48.5 / 61.0 / 11.7
    private static final String[] EMBEDDED_WEIGHTS = {
        "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",
    };
    // END GENERATED WEIGHTS
}
