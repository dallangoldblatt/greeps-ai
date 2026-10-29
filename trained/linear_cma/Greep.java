import greenfoot.*;  // (World, Actor, GreenfootImage, and Greenfoot)

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

/**
 * A Greep controlled by a linear policy.
 *
 * Approach "linear_cma" (policies/linear_cma/; this file is copied to
 * src/Greep.java by `python -m tools.use_policy linear_cma`). The 456
 * weights were found with CMA-ES (policies/linear_cma/train.py), playing whole
 * episodes in the real scenario and scoring each weight vector by its total.
 *
 * Every act round the greep observes its local state, runs one forward pass,
 * and applies all of the free decisions (heading, memory, flags, paint) in one
 * go before deciding whether to make the one real action: move(). Dropping a
 * tomato at the ship and loading tomatoes onto other greeps are reflexes,
 * since both are free and never hurt.
 *
 * The trained weights are embedded below (policies/linear_cma/export.py). A
 * policy.weights file in the working directory (or -Dgreeps.policy=<file>)
 * overrides them; the headless trainer swaps weights via setPolicyWeights().
 */
public class Greep extends Creature {

    /** Which approach this is (policies/<name>/). */
    public static final String POLICY_NAME = "linear_cma";

    // ---- Observation layout (all values in [0, 1] or [-1, 1]) ----
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
    private static final int IN_RANDOM = 22;     // uniform [0, 1)
    public static final int NUM_INPUTS = 23;

    // ---- Output layout ----
    private static final int OUT_MODE = 0;       // 3 scores: keep heading, face home, face pile
    private static final int OUT_TURN = 3;       // extra turn: round(z) steps of 10 degrees, |z| <= 18
    private static final int OUT_MOVE = 4;       // move if > 0
    private static final int OUT_MEMORY = 5;     // 8 bits, bit set if > 0
    private static final int OUT_FLAGS = 13;     // 2 flags, set if > 0
    private static final int OUT_SPIT = 15;      // 4 scores: none, red, orange, purple
    public static final int NUM_OUTPUTS = 19;

    /** One weight row per output: NUM_INPUTS weights followed by a bias. */
    public static final int NUM_PARAMS = NUM_OUTPUTS * (NUM_INPUTS + 1);

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

        switch (argmax(out, OUT_MODE, 3)) {
            case 1:
                turnHome();
                break;
            case 2:
                turnToTomatoes();
                break;
            default:
                break;
        }
        // Linear in z (rather than e.g. 180 * tanh) so that small weight
        // noise rounds to "no turn" instead of making greeps walk in circles.
        double steps = Math.max(-18, Math.min(18, out[OUT_TURN]));
        turn(10 * (int) Math.round(steps));

        int memory = 0;
        for (int bit = 0; bit < 8; bit++) {
            if (out[OUT_MEMORY + bit] > 0) {
                memory |= 1 << bit;
            }
        }
        setMemory(memory);
        setFlag(1, out[OUT_FLAGS] > 0);
        setFlag(2, out[OUT_FLAGS + 1] > 0);

        int spit = argmax(out, OUT_SPIT, 4);
        if (spit > 0) {
            spit(PAINT_COLORS[spit - 1]);
        }

        if (out[OUT_MOVE] > 0) {
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

        in[IN_RANDOM] = Greenfoot.getRandomNumber(1000) / 1000.0;
        return in;
    }

    private static double[] forward(double[] in) {
        if (weights == null) {
            weights = loadWeights();
        }
        double[] out = new double[NUM_OUTPUTS];
        int w = 0;
        for (int o = 0; o < NUM_OUTPUTS; o++) {
            double sum = 0;
            for (int i = 0; i < NUM_INPUTS; i++) {
                sum += weights[w++] * in[i];
            }
            out[o] = sum + weights[w++];
        }
        return out;
    }

    /** Index (relative to start) of the largest of out[start..start+n). */
    private static int argmax(double[] out, int start, int n) {
        int best = 0;
        for (int i = 1; i < n; i++) {
            if (out[start + i] > out[start + best]) {
                best = i;
            }
        }
        return best;
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

    /** The embedded (trained) policy; the hand-written one if none is embedded. */
    public static double[] defaultPolicyWeights() {
        StringBuilder encoded = new StringBuilder();
        for (String chunk : EMBEDDED_WEIGHTS) {
            encoded.append(chunk);
        }
        if (encoded.length() == 0) {
            return initialPolicyWeights();
        }
        DoubleBuffer doubles = ByteBuffer.wrap(Base64.getDecoder().decode(encoded.toString()))
            .order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer();
        if (doubles.remaining() != NUM_PARAMS) {
            throw new IllegalStateException("Embedded policy has " + doubles.remaining()
                + " weights, expected " + NUM_PARAMS);
        }
        double[] w = new double[NUM_PARAMS];
        doubles.get(w);
        return w;
    }

    /**
     * A hand-written policy close to the original heuristic Greep: head home
     * when carrying, face the pile when on one, turn 70 degrees at water and
     * 180 at the world edge, always move. The CMA-ES starting point.
     */
    public static double[] initialPolicyWeights() {
        double[] w = new double[NUM_PARAMS];
        // Face home when carrying; face the pile when on one and empty-handed;
        // otherwise keep heading. Margins of at least 1 between the scores,
        // so that small weight noise does not change the decision.
        setBias(w, OUT_MODE, 1.0);
        set(w, OUT_MODE + 1, IN_CARRYING, 3.0);
        set(w, OUT_MODE + 2, IN_AT_FOOD, 2.0);
        set(w, OUT_MODE + 2, IN_CARRYING, -3.0);
        // In steps of 10 degrees.
        set(w, OUT_TURN, IN_AT_WATER, 7.0);
        set(w, OUT_TURN, IN_AT_EDGE, 18.0);
        setBias(w, OUT_MOVE, 2.0);
        for (int bit = 0; bit < 8; bit++) {
            setBias(w, OUT_MEMORY + bit, -1.0);
        }
        setBias(w, OUT_FLAGS, -1.0);
        setBias(w, OUT_FLAGS + 1, -1.0);
        // A wide margin: random spitting is useless and slows the sim down.
        setBias(w, OUT_SPIT, 3.0);
        return w;
    }

    /**
     * Relative exploration scale for each weight, for the trainer's initial
     * search distribution. Inputs that are usually 0 (events, memory, flags)
     * get full scale: noise there only matters when the event happens.
     * Always-on inputs (bias, heading, home bearing, random) get 0.1, since
     * noise there changes behaviour everywhere at once.
     */
    public static double[] policySearchScales() {
        double[] scales = new double[NUM_PARAMS];
        for (int o = 0; o < NUM_OUTPUTS; o++) {
            for (int i = 0; i <= NUM_INPUTS; i++) {
                boolean dense = i == NUM_INPUTS || i >= IN_HEADING;
                double scale = dense ? 0.1 : 1.0;
                if (o == OUT_TURN) {
                    scale *= 2.0;   // turn is in 10 degree steps
                }
                scales[o * (NUM_INPUTS + 1) + i] = scale;
            }
        }
        return scales;
    }

    private static void set(double[] w, int output, int input, double value) {
        w[output * (NUM_INPUTS + 1) + input] = value;
    }

    private static void setBias(double[] w, int output, double value) {
        w[output * (NUM_INPUTS + 1) + NUM_INPUTS] = value;
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
    // CMA-ES run run1, generation 135, validation 58.9 / 62.8 / 21.8
    private static final String[] EMBEDDED_WEIGHTS = {
        "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",
    };
    // END GENERATED WEIGHTS
}
