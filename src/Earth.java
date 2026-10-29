import greenfoot.*;  // (World, Actor, GreenfootImage, and Greenfoot)
import java.io.PrintWriter;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Random;

/**
 * This is Earth. Or at least some remote, uninhabited part of Earth. Here, Greeps can
 * land and look for piles of tomatoes...
 * 
 * @author Michael Kolling
 * @version 1.0
 */
public class Earth extends World
{
    public static final int RESOLUTION = 1;
    public static final int WIDTH = 800;
    public static final int HEIGHT = 600;

    public static final int SCORE_DISPLAY_TIME = 240;

    private GreenfootImage map;
    private Ship ship;
    private int currentMap;
    
    private int[][][] mapData = {
        { {480, 100, 0}, {40, 721, 532}, {12, 400, 560}, {40, 615, 400},    // map 1
          {40, 642, 192}, {16, 128, 113}, {30, 400, 40} },
 
        { {496, 709, 0}, {10, 322, 422}, {40, 700, 241}, {40, 681, 49},     // map 2
          {10, 317, 54}, {50, 90, 174}, {40, 60, 339} },
          
        { {272, 394, 0}, {10, 39, 30}, {30, 71, 476}, {50, 398, 520},       // map 3
          {40, 655, 492} },          
    };

    // Training/testing only: custom maps from tools/map_editor.py (see
    // -Dgreeps.maps); null means the three maps above.
    private String[] mapImages;

    private int[] scores;
    private boolean finished = false;

    // Training/testing only: per-round state dump (see -Dgreeps.trace)
    private PrintWriter trace;
    private int round = 0;

    /**
     * Create a new world.
     */
    public Earth() {
        super(WIDTH / RESOLUTION, HEIGHT / RESOLUTION, RESOLUTION);

        // Training only: make Greenfoot's random numbers reproducible.
        // Must happen before any creature is created.
        String seed = System.getProperty("greeps.seed");
        if (seed != null) {
            seedGreenfootRandom(Long.parseLong(seed.trim()));
        }

        String tracePath = System.getProperty("greeps.trace");
        if (tracePath != null) {
            try {
                trace = new PrintWriter(tracePath, "UTF-8");
            } catch (IOException e) {
                throw new IllegalStateException("Cannot open trace " + tracePath, e);
            }
        }

        String customMaps = System.getProperty("greeps.maps");
        if (customMaps != null && !customMaps.trim().isEmpty()) {
            loadCustomMaps(expandMapPaths(customMaps.split(",")));
        }

        currentMap = 0;
        scores = new int[mapData.length];
        showMap(currentMap);

        // Training only: start automatically at maximum speed.
        if (Boolean.getBoolean("greeps.training")) {
            Greenfoot.setSpeed(100);
            Greenfoot.start();
        }
    }

    /**
     * Greenfoot calls this at the start of every act round, before any actor
     * acts, so each trace line is the state left by the previous round.
     */
    public void act()
    {
        if (trace != null) {
            writeTraceLine();
        }
        round++;
    }

    /**
     * True once all maps have been played.
     */
    public boolean isFinished()
    {
        return finished;
    }

    /**
     * The per-map scores (only complete once isFinished() is true).
     */
    public int[] getScores()
    {
        return scores.clone();
    }

    private static void seedGreenfootRandom(long seed)
    {
        try {
            Field field = Greenfoot.class.getDeclaredField("randomGenerator");
            field.setAccessible(true);
            ((Random) field.get(null)).setSeed(seed);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Cannot seed Greenfoot random", e);
        }
    }

    /**
     * Read map descriptions written by tools/map_editor.py:
     *   image <file>.png   (800x600, relative to the .map file)
     *   ship <x> <y>
     *   pile <tomatoes> <x> <y>   (one or more)
     */
    /**
     * Map paths as given in -Dgreeps.maps: .map files, or folders standing for
     * the .map files directly inside them (sorted by name).
     */
    private static String[] expandMapPaths(String[] paths)
    {
        java.util.List<String> files = new java.util.ArrayList<>();
        for (String path : paths) {
            java.io.File file = new java.io.File(path.trim());
            if (file.isDirectory()) {
                java.io.File[] maps = file.listFiles((dir, name) -> name.endsWith(".map"));
                java.util.Arrays.sort(maps);
                for (java.io.File map : maps) {
                    files.add(map.getPath());
                }
            }
            else {
                files.add(file.getPath());
            }
        }
        if (files.isEmpty()) {
            throw new IllegalArgumentException("No .map files in " + String.join(",", paths));
        }
        return files.toArray(new String[0]);
    }

    private void loadCustomMaps(String[] files)
    {
        mapData = new int[files.length][][];
        mapImages = new String[files.length];
        for (int m = 0; m < files.length; m++) {
            java.io.File file = new java.io.File(files[m].trim());
            java.util.List<int[]> rows = new java.util.ArrayList<>();
            int[] ship = null;
            try {
                for (String line : java.nio.file.Files.readAllLines(file.toPath())) {
                    int hash = line.indexOf('#');
                    String[] parts = (hash >= 0 ? line.substring(0, hash) : line).trim().split("\\s+");
                    if (parts[0].isEmpty()) {
                        continue;
                    }
                    switch (parts[0]) {
                        case "image":
                            mapImages[m] = new java.io.File(file.getParentFile(), parts[1]).getPath();
                            break;
                        case "ship":   // Earth's layout: {landing y, x, start y}
                            ship = new int[] { Integer.parseInt(parts[2]), Integer.parseInt(parts[1]), 0 };
                            break;
                        case "pile":
                            rows.add(new int[] { Integer.parseInt(parts[1]),
                                Integer.parseInt(parts[2]), Integer.parseInt(parts[3]) });
                            break;
                        default:
                            throw new IllegalArgumentException("Unknown line in " + file + ": " + line);
                    }
                }
            } catch (IOException e) {
                throw new IllegalStateException("Cannot read map " + file, e);
            }
            if (ship == null || mapImages[m] == null || rows.isEmpty()) {
                throw new IllegalStateException(file + " needs an image, a ship and a pile");
            }
            rows.add(0, ship);
            mapData[m] = rows.toArray(new int[0][]);
        }
    }

    private GreenfootImage loadMapImage(int mapNo)
    {
        if (mapImages == null) {
            return new GreenfootImage("map" + mapNo + ".jpg");
        }
        try {
            java.awt.image.BufferedImage source =
                javax.imageio.ImageIO.read(new java.io.File(mapImages[mapNo]));
            if (source.getWidth() != WIDTH || source.getHeight() != HEIGHT) {
                throw new IllegalStateException(mapImages[mapNo] + " must be "
                    + WIDTH + "x" + HEIGHT);
            }
            GreenfootImage image = new GreenfootImage(WIDTH, HEIGHT);
            java.awt.Graphics2D g = image.getAwtImage().createGraphics();
            g.drawImage(source, 0, 0, null);
            g.dispose();
            return image;
        } catch (IOException e) {
            throw new IllegalStateException("Cannot read " + mapImages[mapNo], e);
        }
    }

    private void writeTraceLine()
    {
        StringBuilder line = new StringBuilder();
        line.append("R ").append(round)
            .append(" M ").append(currentMap + 1)
            .append(" S ").append(ship == null ? -1 : ship.getTomatoCount())
            .append(" P ").append(getObjects(TomatoPile.class).size())
            .append(" D ").append(getObjects(Paint.class).size())
            .append(" G");
        for (Greep greep : getObjects(Greep.class)) {
            line.append(' ')
                .append(greep.getX()).append(',')
                .append(greep.getY()).append(',')
                .append(greep.getRotation()).append(',')
                .append(greep.carryingTomato() ? 1 : 0).append(',')
                .append(greep.getMemory()).append(',')
                .append(greep.getFlag(1) ? 1 : 0)
                .append(greep.getFlag(2) ? 1 : 0).append(',')
                .append(greep.getImage().getWidth()).append('x')
                .append(greep.getImage().getHeight());
        }
        trace.println(line);
    }
    
    /**
     * Return true, if the specified coordinate shows water.
     * (Water is defined as a predominantly blueish color.)
     */
    public boolean isWater(int x, int y)
    {
        Color col = map.getColorAt(x, y);
        return col.getBlue() > (col.getRed() * 2);
    }
    
    /**
     * Jump to the given map number (1..n).
     */
    public void jumpToMap(int map)
    {
        clearWorld();
        currentMap = map-1;
        showMap(currentMap);
    }
    
    /**
     * Set up the start scene.
     */
    private void showMap(int mapNo)
    {
        map = loadMapImage(mapNo);
        setBackground(map);
        Counter mapTitle = new Counter(Greep.getAuthorName() + " - Map ", mapNo+1);
        addObject(mapTitle, 200, 20);
        int[][] thisMap = mapData[mapNo];
        for(int i = 1; i < thisMap.length; i++) {
            int[] data = thisMap[i];
            addObject(new TomatoPile(data[0]), data[1], data[2]);
        }
        int[] shipData = thisMap[0];
        ship = new Ship(shipData[0]);
        addObject(ship, shipData[1], shipData[2]);
    }
    
    /**
     * Game is over. Stop running, display score.
     */
    public void mapFinished(int time)
    {
        displayScore(time);
        wait(SCORE_DISPLAY_TIME);
        clearWorld();
        currentMap++;
        if(currentMap < mapData.length) {
            showMap(currentMap);
        }
        else {
            displayFinalScore();
            Greenfoot.stop();
        }
    }

    private void displayScore(int time)
    {
        int points = ship.getTomatoCount() + time;
        scores[currentMap] = points;
        if (trace != null) {
            trace.println("END R " + round + " M " + (currentMap + 1)
                + " tomatoes " + ship.getTomatoCount()
                + " time " + time + " points " + points);
        }
        ScoreBoard board = new ScoreBoard(Greep.getAuthorName(), "Map " + (currentMap+1), "Score: ", currentMap, scores);
        addObject(board, getWidth() / 2, getHeight() / 2);
    }
    
    private void displayFinalScore() {
        clearWorld();
        finished = true;

        // Close the trace before publishing scores: the score file is the
        // signal to the training scripts that the run is complete.
        if (trace != null) {
            trace.close();
            trace = null;
        }

        // Publish the episode scores for policy training
        // (the headless runner reads getScores() instead).
        if (!Boolean.getBoolean("greeps.headless")) {
            try (PrintWriter out =
                    new PrintWriter("training_score.txt")) {
                for (int score : scores) {
                    out.println(score);
                }
                out.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    
        ScoreBoard board = new ScoreBoard(
            Greep.getAuthorName(),
            "Final score",
            "Total: ",
            scores
        );
    
        addObject(board, getWidth() / 2, getHeight() / 2);
    }
    
    private void clearWorld()
    {
        removeObjects(getObjects(null));
    }
    
    private void wait(int time)
    {
        for (int i = 0; i < time; i++) {
            Greenfoot.delay(1);
        }
    }
}