package live2d.automation;

import java.util.concurrent.atomic.AtomicInteger;
import javax.swing.*;
import javax.swing.tree.DefaultMutableTreeNode;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

/** Exercises real Swing widgets without launching Cubism or a desktop window. */
@SuppressWarnings("unchecked")
public final class SwingBridgeTest {
    /** Synthetic windows are allocated without constructors and never create native peers. */
    public static final class GuardFrame extends java.awt.Frame {
        String title;
        @Override public String getTitle() { return title; }
        @Override public String getName() { return "test-frame"; }
        @Override public boolean isShowing() { return true; }
        @Override public boolean isVisible() { return true; }
        @Override public boolean isEnabled() { return true; }
        @Override public java.awt.Rectangle getBounds() { return new java.awt.Rectangle(0, 0, 600, 400); }
        @Override public java.awt.Point getLocationOnScreen() { return new java.awt.Point(); }
        @Override public java.awt.Component[] getComponents() { return new java.awt.Component[0]; }
    }
    public static final class GuardDialog extends java.awt.Dialog {
        GuardDialog() { super((java.awt.Frame)null); }
        @Override public String getTitle() { return "test-dialog"; }
        @Override public String getName() { return "test-dialog"; }
        @Override public boolean isShowing() { return true; }
        @Override public boolean isVisible() { return true; }
        @Override public boolean isEnabled() { return true; }
        @Override public boolean isModal() { return true; }
        @Override public java.awt.Rectangle getBounds() { return new java.awt.Rectangle(0, 0, 300, 200); }
        @Override public java.awt.Point getLocationOnScreen() { return new java.awt.Point(); }
        @Override public java.awt.Component[] getComponents() { return new java.awt.Component[0]; }
    }
    public static final class GuardButton extends JToggleButton {
        java.awt.Container window;
        @Override public java.awt.Container getParent() { return window; }
    }

    private static java.lang.reflect.Field field(Class<?> owner, String name) throws Exception {
        java.lang.reflect.Field field = owner.getDeclaredField(name); field.setAccessible(true); return field;
    }

    private static <T extends java.awt.Window> T window(Class<T> type) throws Exception {
        Class<?> unsafeClass = Class.forName("sun.misc.Unsafe");
        Object unsafe = field(unsafeClass, "theUnsafe").get(null);
        T window = type.cast(unsafeClass.getMethod("allocateInstance", Class.class).invoke(unsafe, type));
        java.lang.reflect.Field context = field(java.awt.Component.class, "appContext");
        context.set(window, context.get(new JPanel()));
        field(java.awt.Window.class, "weakThis").set(window, new java.lang.ref.WeakReference<>(window));
        java.lang.reflect.Method add = java.awt.Window.class.getDeclaredMethod("addToWindowList");
        add.setAccessible(true); add.invoke(window);
        return window;
    }

    private static void unregister(java.awt.Window... windows) throws Exception {
        java.lang.reflect.Method remove = java.awt.Window.class.getDeclaredMethod("removeFromWindowList");
        remove.setAccessible(true);
        for (java.awt.Window window : windows) remove.invoke(window);
    }

    private static JSONObject guardedClick(java.awt.Window owner, String document) {
        GuardButton button = new GuardButton(); button.window = owner;
        JSONObject request = action("click", SwingBridge.snapshot(button, 0).get("id"));
        request.put("expected_document", document);
        SwingBridge.perform(request);
        if (!button.isSelected()) throw new AssertionError("Allowed guarded button did not execute");
        return request;
    }

    private static void checkFullDocumentName() throws Exception {
        GuardFrame main = window(GuardFrame.class); main.title = "Cubism - Draft - Avatar.cmo3";
        try {
            try { guardedClick(main, "Avatar.cmo3"); throw new AssertionError("Short filename changed another document"); }
            catch (IllegalArgumentException expected) { }
            guardedClick(main, "Draft - Avatar.cmo3");
        } finally { unregister(main); }
    }

    private static void checkActualModalBlocker() throws Exception {
        GuardFrame main = window(GuardFrame.class); main.title = "Cubism - Avatar.cmo3";
        GuardDialog earlier = window(GuardDialog.class), later = window(GuardDialog.class);
        java.lang.reflect.Field blocker = field(java.awt.Window.class, "modalBlocker");
        try {
            blocker.set(main, later); blocker.set(later, earlier);
            guardedClick(earlier, "Avatar.cmo3");
            try { guardedClick(later, "Avatar.cmo3"); throw new AssertionError("Blocked modal was actionable"); }
            catch (IllegalArgumentException expected) { }
            JSONObject active = SwingBridge.snapshot(earlier, 0), blocked = SwingBridge.snapshot(later, 0);
            if (!active.containsKey("blocked_by") || active.get("blocked_by") != null
                || !active.get("id").equals(blocked.get("blocked_by")))
                throw new AssertionError("Window snapshots lost the actual modal blocker");
            blocker.set(later, null); blocker.set(earlier, later);
            guardedClick(later, "Avatar.cmo3");
            try { guardedClick(earlier, "Avatar.cmo3"); throw new AssertionError("Newly blocked modal was actionable"); }
            catch (IllegalArgumentException expected) { }
        } finally { unregister(main, earlier, later); }
    }

    /** Loads independent copies of the bridge, as successive attachments do. */
    private static final class AttachmentLoader extends ClassLoader {
        AttachmentLoader() { super(SwingBridge.class.getClassLoader()); }
        @Override protected synchronized Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
            if (!name.equals("live2d.automation.SwingBridge") && !name.startsWith("live2d.automation.SwingBridge$"))
                return super.loadClass(name, resolve);
            Class<?> type = findLoadedClass(name);
            if (type == null) {
                try (java.io.InputStream stream = getParent().getResourceAsStream(name.replace('.', '/') + ".class")) {
                    byte[] bytes = stream.readAllBytes(); type = defineClass(name, bytes, 0, bytes.length);
                } catch (java.io.IOException error) { throw new ClassNotFoundException(name, error); }
            }
            if (resolve) resolveClass(type);
            return type;
        }
    }

    private static void checkAttachmentIds() throws Exception {
        Class<?> first = new AttachmentLoader().loadClass("live2d.automation.SwingBridge");
        Class<?> second = new AttachmentLoader().loadClass("live2d.automation.SwingBridge");
        JButton old = new JButton("old"); JToggleButton replacement = new JToggleButton("new");
        Object oldId = ((JSONObject)first.getMethod("snapshot", java.awt.Component.class, int.class).invoke(null, old, 0)).get("id");
        Object newId = ((JSONObject)second.getMethod("snapshot", java.awt.Component.class, int.class).invoke(null, replacement, 0)).get("id");
        try {
            second.getMethod("perform", JSONObject.class).invoke(null, action("click", oldId));
            throw new AssertionError("Previous attachment ID targeted a replacement widget");
        } catch (java.lang.reflect.InvocationTargetException error) {
            if (!(error.getCause() instanceof IllegalArgumentException)) throw error;
        }
        if (replacement.isSelected()) throw new AssertionError("Stale ID changed the replacement widget");
        second.getMethod("perform", JSONObject.class).invoke(null, action("click", newId));
        if (!replacement.isSelected()) throw new AssertionError("Current attachment ID was rejected");
    }

    private static void checkSecurityGuards() throws Exception {
        StringBuilder failures = new StringBuilder();
        for (String check : new String[]{"document", "attachment", "modal"}) {
            try {
                if (check.equals("document")) checkFullDocumentName();
                if (check.equals("attachment")) checkAttachmentIds();
                if (check.equals("modal")) checkActualModalBlocker();
            } catch (Throwable error) { failures.append(check).append(": ").append(error).append('\n'); }
        }
        if (failures.length() > 0) throw new AssertionError(failures.toString());
    }
    public interface NumericListener extends java.util.EventListener {
        void a(javax.swing.event.ChangeEvent event, Object value, boolean adjusting, boolean mixed);
    }
    public static final class IntegerFixture extends JPanel {
        int value;
        final javax.swing.event.EventListenerList callbacks = new javax.swing.event.EventListenerList();
        public void a(int next, boolean clamp) { value = next; }
        public javax.swing.event.EventListenerList C() { return callbacks; }
    }
    public static final class NumericFixture extends JPanel {
        float value;
        float committed = -1;
        public void a(float next, boolean clamp) { value = next; }
        public void a(float next) { committed = next; }
    }
    public static final class FocusFixture extends JPanel {
        boolean accepted;
        int requests;
        @Override public boolean requestFocusInWindow() {
            requests++;
            return accepted;
        }
    }
    private static JSONObject action(String name, Object id) {
        JSONObject value = new JSONObject();
        value.put("action", name);
        value.put("id", id);
        return value;
    }

    private static void rejects(JSONObject request) {
        try {
            SwingBridge.perform(request);
            throw new AssertionError("Locked widget accepted " + request.get("action"));
        } catch (IllegalArgumentException expected) { }
    }

    private static void checkLockedWidgets() {
        JTextField text = new JTextField("before");
        AtomicInteger commits = new AtomicInteger();
        text.addActionListener(event -> commits.incrementAndGet());
        JSONObject edit = action("set_text", SwingBridge.snapshot(text, 0).get("id"));
        edit.put("value", "after"); edit.put("enter", true);
        text.setEditable(false); rejects(edit);
        text.setEditable(true); text.setEnabled(false); rejects(edit);
        if (!text.getText().equals("before") || commits.get() != 0)
            throw new AssertionError("Locked text changed or committed");

        JComboBox<String> combo = new JComboBox<>(new String[]{"first", "second"});
        JSONObject choose = action("select_combo", SwingBridge.snapshot(combo, 0).get("id"));
        choose.put("index", 1L); combo.setEnabled(false); rejects(choose);
        if (combo.getSelectedIndex() != 0) throw new AssertionError("Disabled combo changed");

        JTree tree = new JTree(); tree.setSelectionRow(0);
        JSONObject branch = action("select_tree", SwingBridge.snapshot(tree, 0).get("id"));
        JSONArray path = new JSONArray(); path.add(0L);
        JSONArray paths = new JSONArray(); paths.add(path); branch.put("paths", paths);
        tree.setEnabled(false); rejects(branch);
        if (tree.getSelectionRows()[0] != 0) throw new AssertionError("Disabled tree changed");

        JList<String> list = new JList<>(new String[]{"first", "second"}); list.setSelectedIndex(0);
        JSONObject item = action("select_list", SwingBridge.snapshot(list, 0).get("id"));
        item.put("index", 1L); list.setEnabled(false); rejects(item);
        if (list.getSelectedIndex() != 0) throw new AssertionError("Disabled list changed");

        JTable table = new JTable(2, 1); table.setRowSelectionInterval(0, 0);
        JSONObject row = action("select_table", SwingBridge.snapshot(table, 0).get("id"));
        JSONArray rows = new JSONArray(); rows.add(1L); row.put("rows", rows);
        table.setEnabled(false); rejects(row);
        if (table.getSelectedRow() != 0) throw new AssertionError("Disabled table changed");

        JPanel panel = new JPanel();
        panel.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent event) { commits.incrementAndGet(); }
        });
        Object panelId = SwingBridge.snapshot(panel, 0).get("id"); panel.setEnabled(false);
        for (String name : new String[]{"mouse", "drag"}) {
            JSONObject gesture = action(name, panelId);
            gesture.put("x", 1L); gesture.put("y", 1L);
            gesture.put("to_x", 2L); gesture.put("to_y", 2L); rejects(gesture);
        }
        if (commits.get() != 0) throw new AssertionError("Disabled pointer target received a press");
    }

    private static java.lang.ref.WeakReference<JPanel> registerDisposable(Object[] id) {
        JPanel panel = new JPanel();
        id[0] = SwingBridge.snapshot(panel, 0).get("id");
        return new java.lang.ref.WeakReference<>(panel);
    }

    private static void checkRegistryReleasesWidgets() throws Exception {
        JPanel live = new JPanel();
        Object liveId = SwingBridge.snapshot(live, 0).get("id");
        Object[] expiredId = new Object[1];
        java.lang.ref.WeakReference<JPanel> disposable = registerDisposable(expiredId);
        for (int attempt = 0; attempt < 30 && disposable.get() != null; attempt++) {
            System.gc(); Thread.sleep(20);
        }
        if (disposable.get() != null) throw new AssertionError("Registry retains a discarded widget");
        if (!liveId.equals(SwingBridge.snapshot(live, 0).get("id")))
            throw new AssertionError("Live widget lost its stable id");
        try {
            SwingBridge.perform(action("focus", expiredId[0]));
            throw new AssertionError("Collected widget id was accepted");
        } catch (IllegalArgumentException expected) {
            if (!expected.getMessage().contains("expired")) throw expected;
        }
    }

    private static void checkLargeTableLookup() {
        Object[][] data = new Object[320][3];
        for (int row = 0; row < data.length; row++) data[row][2] = "Part " + row;
        JTable table = new JTable(data, new Object[]{"Visible", "Locked", "Name"});
        table.setRowSelectionInterval(3, 3);
        JSONObject find = action("find_table_rows", SwingBridge.snapshot(table, 0).get("id"));
        find.put("column", 2L);
        JSONArray names = new JSONArray(); names.add("Part 150"); names.add("Part 319");
        find.put("values", names);
        JSONArray rows = (JSONArray)SwingBridge.perform(find).get("rows");
        if (rows.size() != 2 || ((Number)rows.get(0)).intValue() != 150 || ((Number)rows.get(1)).intValue() != 319)
            throw new AssertionError("Lookup missed rows beyond the snapshot limit");
        if (table.getSelectedRow() != 3) throw new AssertionError("Row lookup changed selection");
        names.set(0, "Absent"); rejects(find);
        names.set(0, "Part 150"); table.setValueAt("Part 150", 0, 2); rejects(find);
    }

    private static void checkRejectedRowsPreserveSelection() {
        JTable table = new JTable(3, 1);
        AtomicInteger changes = new AtomicInteger();
        table.getSelectionModel().addListSelectionListener(event -> changes.incrementAndGet());
        Object tableId = SwingBridge.snapshot(table, 0).get("id");
        for (Object invalid : new Object[]{8L, -1L, "bad", null, 0.5, 4294967296L}) {
            table.setRowSelectionInterval(1, 1); changes.set(0);
            JSONArray rows = new JSONArray(); rows.add(0L); rows.add(invalid);
            JSONObject request = action("select_table", tableId); request.put("rows", rows);
            try { SwingBridge.perform(request); throw new AssertionError("Invalid row accepted: " + invalid); }
            catch (IllegalArgumentException expected) { }
            if (!java.util.Arrays.equals(table.getSelectedRows(), new int[]{1}) || changes.get() != 0)
                throw new AssertionError("Rejected row list changed the selection or notified listeners");
        }
        for (Object invalid : new Object[]{null, "not a list"}) {
            changes.set(0);
            JSONObject request = action("select_table", tableId); request.put("rows", invalid);
            try { SwingBridge.perform(request); throw new AssertionError("Invalid row container accepted"); }
            catch (IllegalArgumentException expected) { }
            if (table.getSelectedRow() != 1 || changes.get() != 0)
                throw new AssertionError("Invalid row container changed selection");
        }
        JSONArray rows = new JSONArray(); rows.add(2L); rows.add(0L);
        JSONObject request = action("select_table", tableId); request.put("rows", rows);
        SwingBridge.perform(request);
        if (!java.util.Arrays.equals(table.getSelectedRows(), new int[]{0, 2}))
            throw new AssertionError("Valid row list did not replace selection");
        request.put("rows", new JSONArray()); SwingBridge.perform(request);
        if (table.getSelectedRowCount() != 0) throw new AssertionError("Empty row list did not clear selection");
    }

    private static void checkDetailedCaptureFitsTransport() throws Exception {
        JPanel canvas = new JPanel() {
            @Override protected void paintComponent(java.awt.Graphics graphics) {
                java.awt.Graphics2D target = (java.awt.Graphics2D)graphics.create();
                double scale = target.getTransform().getScaleX();
                int size = (int)Math.round(getWidth() * scale);
                java.awt.image.BufferedImage noise = new java.awt.image.BufferedImage(
                    size, size, java.awt.image.BufferedImage.TYPE_INT_ARGB);
                int[] pixels = ((java.awt.image.DataBufferInt)noise.getRaster().getDataBuffer()).getData();
                java.util.Random random = new java.util.Random(42);
                for (int i = 0; i < pixels.length; i++) pixels[i] = random.nextInt();
                target.scale(1 / scale, 1 / scale);
                target.drawImage(noise, 0, 0, null); target.dispose();
            }
        };
        canvas.setSize(1536, 1536);
        JSONObject capture = SwingBridge.perform(action("capture", SwingBridge.snapshot(canvas, 0).get("id")));
        capture.put("request_id", "r".repeat(80));
        if (capture.toJSONString().getBytes(java.nio.charset.StandardCharsets.UTF_8).length > 32 * 1024 * 1024)
            throw new AssertionError("Detailed capture exceeds the Python transport response limit");
        byte[] png = java.util.Base64.getDecoder().decode((String)capture.get("png_base64"));
        java.awt.image.BufferedImage decoded = javax.imageio.ImageIO.read(new java.io.ByteArrayInputStream(png));
        if (decoded == null || decoded.getWidth() != decoded.getHeight() || decoded.getWidth() < 1)
            throw new AssertionError("Capture is not a valid proportional image");
    }

    private static void checkHoverBeforeGesture(String actionName) {
        JPanel panel = new JPanel();
        AtomicInteger hoveredX = new AtomicInteger(-1);
        AtomicInteger hoveredY = new AtomicInteger(-1);
        AtomicInteger pressedX = new AtomicInteger(-1);
        AtomicInteger draggedX = new AtomicInteger(-1);
        java.awt.event.MouseAdapter receiver = new java.awt.event.MouseAdapter() {
            public void mouseMoved(java.awt.event.MouseEvent event) {
                if (event.getButton() != java.awt.event.MouseEvent.NOBUTTON || event.getClickCount() != 0
                    || event.getModifiersEx() != java.awt.event.MouseEvent.SHIFT_DOWN_MASK)
                    throw new AssertionError("Hover event has incorrect button, count, or modifiers");
                hoveredX.set(event.getX());
                hoveredY.set(event.getY());
            }
            public void mousePressed(java.awt.event.MouseEvent event) {
                if (hoveredX.get() == event.getX() && hoveredY.get() == event.getY())
                    pressedX.set(event.getX());
            }
            public void mouseDragged(java.awt.event.MouseEvent event) {
                if (pressedX.get() >= 0) draggedX.set(event.getX());
            }
        };
        panel.addMouseListener(receiver);
        panel.addMouseMotionListener(receiver);
        JSONObject gesture = action(actionName, SwingBridge.snapshot(panel, 0).get("id"));
        gesture.put("x", 12L); gesture.put("y", 8L);
        gesture.put("modifiers", java.awt.event.MouseEvent.SHIFT_DOWN_MASK);
        gesture.put("to_x", 16L); gesture.put("to_y", 8L);
        SwingBridge.perform(gesture);
        if (pressedX.get() != 12) throw new AssertionError(actionName + " pressed before establishing hover at its start");
        if (actionName.equals("drag") && draggedX.get() != 16)
            throw new AssertionError("Hover-gated drag did not reach its endpoint");
    }

    public static void main(String[] args) throws Exception {
        checkRejectedRowsPreserveSelection();
        checkSecurityGuards();
        checkLockedWidgets();
        checkRegistryReleasesWidgets();
        checkLargeTableLookup();
        checkDetailedCaptureFitsTransport();
        java.nio.file.Path directory = java.nio.file.Files.createTempDirectory("live2d-bridge-test-");
        java.nio.file.Files.setPosixFilePermissions(directory, java.nio.file.attribute.PosixFilePermissions.fromString("rwx------"));
        java.nio.file.Path outside = java.nio.file.Files.createTempFile("live2d-outside-", ".txt");
        java.nio.file.Files.writeString(outside, "untouched");
        java.nio.file.Path link = directory.resolve("unsafe");
        java.nio.file.Files.createSymbolicLink(link, outside);
        try { SwingBridge.writePrivate(link, "changed"); throw new AssertionError("Symlink transport write accepted"); }
        catch (java.io.IOException expected) { }
        if (!java.nio.file.Files.readString(outside).equals("untouched")) throw new AssertionError("External target changed");
        try { SwingBridge.requirePrivate(link, false); throw new AssertionError("Symlink transport read accepted"); }
        catch (java.io.IOException expected) { }
        java.nio.file.Path privateFile = directory.resolve("private.json");
        SwingBridge.writePrivate(privateFile, "{}");
        SwingBridge.requirePrivate(privateFile, false);
        if (!java.nio.file.Files.getPosixFilePermissions(privateFile).equals(java.nio.file.attribute.PosixFilePermissions.fromString("rw-------")))
            throw new AssertionError("Private file permissions were not applied");
        java.nio.file.Files.delete(privateFile); java.nio.file.Files.delete(link);
        java.nio.file.Files.delete(directory); java.nio.file.Files.delete(outside);
        Runnable checks = () -> {
            checkHoverBeforeGesture("mouse");
            checkHoverBeforeGesture("drag");
            FocusFixture focus = new FocusFixture();
            Object focusId = SwingBridge.snapshot(focus, 0).get("id");
            for (boolean accepted : new boolean[]{true, false}) {
                focus.accepted = accepted;
                JSONObject focused = SwingBridge.perform(action("focus", focusId));
                if (!"success".equals(focused.get("status")) || !Boolean.valueOf(accepted).equals(focused.get("focused")))
                    throw new AssertionError("Focus request result was not reported");
            }
            if (focus.requests != 2) throw new AssertionError("Focus was not requested exactly once per action");
            JPanel unattached = new JPanel();
            Object unattachedId = SwingBridge.snapshot(unattached, 0).get("id");
            if (!Boolean.FALSE.equals(SwingBridge.perform(action("focus", unattachedId)).get("focused")))
                throw new AssertionError("Unattached component falsely reported focus acceptance");
            NumericFixture numeric = new NumericFixture();
            SwingBridge.commitFloat(numeric, 35);
            if (numeric.value != 35 || numeric.committed != 35)
                throw new AssertionError("Numeric value did not reach the change listener");
            IntegerFixture integer = new IntegerFixture();
            AtomicInteger integerCommit = new AtomicInteger();
            integer.callbacks.add(NumericListener.class, (event, value, adjusting, mixed) -> integerCommit.set((Integer)value));
            SwingBridge.commitInteger(integer, 575, javax.swing.event.ChangeEvent.class, NumericListener.class);
            if (integer.value != 575 || integerCommit.get() != 575)
                throw new AssertionError("Draw order did not reach the registered model listener");
            JPanel panel = new JPanel();
            JButton button = new JButton("Confirm");
            AtomicInteger clicked = new AtomicInteger();
            button.addActionListener(event -> clicked.incrementAndGet());
            JTextField field = new JTextField("before");
            AtomicInteger committed = new AtomicInteger();
            AtomicInteger focusCommit = new AtomicInteger();
            field.addFocusListener(new java.awt.event.FocusAdapter() {
                public void focusLost(java.awt.event.FocusEvent event) { focusCommit.incrementAndGet(); }
            });
            field.addKeyListener(new java.awt.event.KeyAdapter() {
                public void keyPressed(java.awt.event.KeyEvent event) {
                    if (event.getKeyCode() == java.awt.event.KeyEvent.VK_ENTER) committed.incrementAndGet();
                }
            });
            JComboBox<String> combo = new JComboBox<>(new String[]{"Face", "Mouth", "Hair"});
            DefaultMutableTreeNode root = new DefaultMutableTreeNode("Model");
            DefaultMutableTreeNode face = new DefaultMutableTreeNode("Face");
            DefaultMutableTreeNode mouth = new DefaultMutableTreeNode("Mouth_Inside");
            root.add(face);
            face.add(mouth);
            JTree tree = new JTree(root);
            panel.add(button); panel.add(field); panel.add(combo); panel.add(tree);
            JTable table = new JTable(new Object[][]{{"Mouth_Inside"}, {"Eye_L"}}, new Object[]{"Name"});
            panel.add(table);
            JList<String> list = new JList<>(new String[]{"New model", "Existing model"}); panel.add(list);
            AtomicInteger mouseX = new AtomicInteger(-1);
            panel.addMouseListener(new java.awt.event.MouseAdapter() {
                public void mouseClicked(java.awt.event.MouseEvent event) { mouseX.set(event.getX()); }
            });
            AtomicInteger dragX = new AtomicInteger(-1);
            panel.addMouseMotionListener(new java.awt.event.MouseMotionAdapter() {
                public void mouseDragged(java.awt.event.MouseEvent event) { dragX.set(event.getX()); }
            });

            JSONObject view = SwingBridge.snapshot(panel, 4);
            JSONArray children = (JSONArray)view.get("children");
            if (children == null || children.size() < 4) throw new AssertionError("Missing widget tree");
            Object buttonId = ((JSONObject)children.get(0)).get("id");
            Object fieldId = ((JSONObject)children.get(1)).get("id");
            JPanel menuOwner = new JPanel();
            JMenuBar bar = new JMenuBar();
            JMenu file = new JMenu("File"), nested = new JMenu("Export");
            JMenuItem item = new JMenuItem("Model");
            menuOwner.add(bar); bar.add(file); file.add(nested); nested.add(item);
            if (!SwingBridge.isOwnedBy(item, menuOwner))
                throw new AssertionError("Hidden submenu lost its owning document");
            if (SwingBridge.isOwnedBy(item, new JPanel()))
                throw new AssertionError("Unrelated window claimed a submenu");
            Object comboId = ((JSONObject)children.get(2)).get("id");
            Object treeId = ((JSONObject)children.get(3)).get("id");
            JSONObject guarded = action("click", buttonId);
            guarded.put("expected_document", "portrait.cmo3");
            try { SwingBridge.perform(guarded); throw new AssertionError("Document guard accepted an unattached widget"); }
            catch (IllegalArgumentException expected) { }
            if (clicked.get() != 0) throw new AssertionError("Document guard ran after the click");
            try {
                SwingBridge.perform(action("close_dialog", buttonId));
                throw new AssertionError("Dialog close accepted a non-window widget");
            } catch (IllegalArgumentException expected) {
                if (!expected.getMessage().contains("dialog")) throw expected;
            }
            SwingBridge.perform(action("click", buttonId));
            if (clicked.get() != 1) throw new AssertionError("Button did not execute on Swing");

            JSONObject text = action("set_text", fieldId); text.put("value", "Eye_L");
            SwingBridge.perform(text);
            if (!field.getText().equals("Eye_L")) throw new AssertionError("Text field was not updated");
            text.put("enter", true); SwingBridge.perform(text);
            if (committed.get() != 1) throw new AssertionError("Inline numeric editor was not committed");
            if (focusCommit.get() != 1) throw new AssertionError("Inline numeric editor did not finalize focus");

            JSONObject select = action("select_combo", comboId); select.put("value", "Mouth");
            SwingBridge.perform(select);
            if (!combo.getSelectedItem().equals("Mouth")) throw new AssertionError("Wrong combo item");
            select.put("value", "Missing");
            try { SwingBridge.perform(select); throw new AssertionError("Unknown combo item accepted"); }
            catch (IllegalArgumentException expected) { }

            JSONObject treeSelect = action("select_tree", treeId);
            JSONArray paths = new JSONArray(); JSONArray path = new JSONArray();
            path.add(0L); path.add(0L); paths.add(path); treeSelect.put("paths", paths);
            SwingBridge.perform(treeSelect);
            if (tree.getLastSelectedPathComponent() != mouth) throw new AssertionError("Wrong tree object");
            Object tableId = ((JSONObject)children.get(4)).get("id");
            JSONObject tableSelect = action("select_table", tableId);
            JSONArray rows = new JSONArray(); rows.add(1L); tableSelect.put("rows", rows);
            SwingBridge.perform(tableSelect);
            if (table.getSelectedRow() != 1) throw new AssertionError("Wrong table row");
            JSONObject scrollRow = action("scroll_table_row", tableId);
            scrollRow.put("row", 0L); SwingBridge.perform(scrollRow);
            if (table.getSelectedRow() != 1) throw new AssertionError("Scrolling changed selection");
            JSONObject mouse = action("mouse", view.get("id")); mouse.put("x", 12L); mouse.put("y", 8L);
            SwingBridge.perform(mouse);
            if (mouseX.get() != 12) throw new AssertionError("Mouse event was not delivered");
            JSONObject listSelect = action("select_list", ((JSONObject)children.get(5)).get("id"));
            listSelect.put("index", 1L); SwingBridge.perform(listSelect);
            if (list.getSelectedIndex() != 1) throw new AssertionError("Wrong list selection");
            JSONObject drag = action("drag", view.get("id"));
            drag.put("x", 10L); drag.put("y", 10L); drag.put("to_x", 30L); drag.put("to_y", 12L);
            SwingBridge.perform(drag);
            if (dragX.get() != 30) throw new AssertionError("Drag did not reach the requested endpoint");
            panel.setSize(50, 30); panel.removeAll(); panel.setBackground(java.awt.Color.RED);
            JSONObject capture = SwingBridge.perform(action("capture", view.get("id")));
            if (!capture.containsKey("png_base64")) throw new AssertionError("Missing component image");
            System.out.println("Swing widget selection, text, combo, and button checks passed");
        };
        checks.run();
    }
}
