package live2d.automation;

import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.WindowEvent;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.lang.instrument.Instrumentation;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.nio.file.attribute.*;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.swing.*;
import javax.swing.text.JTextComponent;
import javax.swing.tree.*;
import javax.imageio.ImageIO;
import org.json.simple.*;
import org.json.simple.parser.JSONParser;

/** Operates editor widgets on Swing's event thread without changing model file formats. */
@SuppressWarnings({"unchecked", "rawtypes"})
public final class SwingBridge {
    private static final String attachment = UUID.randomUUID().toString();
    private static final Map<Component, String> ids = new WeakHashMap<>();
    private static final Map<String, WidgetReference> components = new HashMap<>();
    private static final ReferenceQueue<Component> expired = new ReferenceQueue<>();
    private static final int MAX_RESPONSE_BYTES = 32 * 1024 * 1024;
    private static final AtomicBoolean running = new AtomicBoolean(false);
    private static long sequence = 0;

    /** Retains an ID for cleanup without keeping its widget alive. */
    private static final class WidgetReference extends WeakReference<Component> {
        final String id;
        WidgetReference(Component component, String id) {
            super(component, expired); this.id = id;
        }
    }

    /** Updates the editor control and emits its public model-change notification. */
    static void commitFloat(Component control, float value) {
        if (!Float.isFinite(value)) throw new IllegalArgumentException("Number must be finite");
        try {
            control.getClass().getMethod("a", float.class, boolean.class).invoke(control, value, true);
            control.getClass().getMethod("a", float.class).invoke(control, value);
        } catch (ReflectiveOperationException error) {
            throw new IllegalStateException("Unsupported Cubism numeric control", error);
        }
    }

    /** Emits the integer widget's public change-listener contract after setting its value. */
    static void commitInteger(Component control, int value, Class<?> eventClass, Class<?> listenerClass) {
        try {
            control.getClass().getMethod("a", int.class, boolean.class).invoke(control, value, true);
            javax.swing.event.EventListenerList listeners = (javax.swing.event.EventListenerList)
                control.getClass().getMethod("C").invoke(control);
            Object event = eventClass.getConstructor(Object.class).newInstance(control);
            java.lang.reflect.Method notify = listenerClass.getMethod("a", eventClass, Object.class, boolean.class, boolean.class);
            Object[] entries = listeners.getListenerList();
            for (int index = 0; index < entries.length; index += 2)
                if (entries[index] == listenerClass) notify.invoke(entries[index + 1], event, Integer.valueOf(value), false, false);
        } catch (ReflectiveOperationException error) {
            throw new IllegalStateException("Unsupported Cubism integer control", error);
        }
    }

    private static String id(Component component) {
        WidgetReference reference;
        while ((reference = (WidgetReference)expired.poll()) != null) components.remove(reference.id);
        return ids.computeIfAbsent(component, key -> {
            String value = attachment + ":" + ++sequence;
            components.put(value, new WidgetReference(key, value)); return value;
        });
    }

    private static Component lookup(Object value) {
        WidgetReference reference = value instanceof String ? components.get(value) : null;
        return reference == null ? null : reference.get();
    }

    /** Reads AWT's blocker instead of inferring modal order from focus or window creation. */
    private static Dialog modalBlocker(Window window) {
        try {
            java.lang.reflect.Method getter = Window.class.getDeclaredMethod("getModalBlocker");
            if (!getter.trySetAccessible()) throw new IllegalStateException("Cannot inspect modal blocking; reattach the bridge");
            return (Dialog)getter.invoke(window);
        } catch (ReflectiveOperationException | SecurityException error) {
            throw new IllegalStateException("Cannot inspect modal blocking", error);
        }
    }

    private static String text(Component component) {
        if (component instanceof JLabel) return ((JLabel)component).getText();
        if (component instanceof AbstractButton) return ((AbstractButton)component).getText();
        if (component instanceof JTextComponent) return ((JTextComponent)component).getText();
        if (component instanceof Frame) return ((Frame)component).getTitle();
        if (component instanceof Dialog) return ((Dialog)component).getTitle();
        return component.getName();
    }

    private static String rendererText(Component component) {
        String own = text(component);
        if (own != null && !own.isEmpty()) return own;
        if (component instanceof Container) {
            StringJoiner label = new StringJoiner(" | ");
            for (Component child : ((Container)component).getComponents()) {
                String childText = rendererText(child);
                if (!childText.isEmpty()) label.add(childText);
            }
            return label.toString();
        }
        return "";
    }

    private static void preparePrint(Component component, ArrayList<Component> panels, double scaleX, double scaleY) throws Exception {
        if (component.getClass().getName().equals("com.jogamp.opengl.awt.GLJPanel")) {
            component.getClass().getMethod("setupPrint", double.class, double.class, int.class, int.class, int.class)
                .invoke(component, scaleX, scaleY, 0, -1, -1);
            panels.add(component);
        }
        if (component instanceof Container)
            for (Component child : ((Container)component).getComponents()) preparePrint(child, panels, scaleX, scaleY);
    }

    private static void treeNodes(JTree tree, Object node, JSONArray path, JSONArray nodes) {
        if (nodes.size() >= 1000) return;
        TreeModel model = tree.getModel();
        JSONObject value = new JSONObject();
        value.put("path", new JSONArray()); ((JSONArray)value.get("path")).addAll(path);
        Component renderer = tree.getCellRenderer().getTreeCellRendererComponent(
            tree, node, false, false, model.isLeaf(node), -1, false);
        String label = rendererText(renderer);
        value.put("label", label.isEmpty() ? String.valueOf(node) : label);
        value.put("class", node.getClass().getName());
        nodes.add(value);
        for (int index = 0; index < model.getChildCount(node); index++) {
            JSONArray childPath = new JSONArray(); childPath.addAll(path); childPath.add(index);
            treeNodes(tree, model.getChild(node, index), childPath, nodes);
        }
    }

    /** Inspects a widget and bounded child data without retaining the widget. */
    public static JSONObject snapshot(Component component, int depth) {
        JSONObject value = new JSONObject();
        value.put("id", id(component)); value.put("class", component.getClass().getName());
        value.put("text", text(component)); value.put("visible", component.isVisible());
        value.put("showing", component.isShowing()); value.put("enabled", component.isEnabled());
        if (component instanceof Window) {
            Dialog blocker = modalBlocker((Window)component);
            value.put("blocked_by", blocker == null ? null : id(blocker));
        }
        if (component instanceof Dialog) value.put("modal", ((Dialog)component).isModal());
        if (component instanceof JComponent) value.put("tooltip", ((JComponent)component).getToolTipText());
        value.put("kind", component instanceof JTextComponent ? "text" : component instanceof AbstractButton ? "button" : component instanceof JTable ? "table" : component instanceof JComboBox ? "combo" : component instanceof JTree ? "tree" : "component");
        Rectangle bounds = component.getBounds();
        if (component.isShowing()) {
            try { Point screen = component.getLocationOnScreen(); bounds.x = screen.x; bounds.y = screen.y; }
            catch (IllegalComponentStateException ignored) { }
        }
        value.put("bounds", Arrays.asList(bounds.x, bounds.y, bounds.width, bounds.height));
        if (component instanceof AbstractButton) value.put("selected", ((AbstractButton)component).isSelected());
        if (component instanceof JComboBox) {
            JComboBox combo = (JComboBox)component; JSONArray options = new JSONArray();
            for (int index = 0; index < combo.getItemCount(); index++) options.add(String.valueOf(combo.getItemAt(index)));
            value.put("options", options); value.put("selected_index", combo.getSelectedIndex());
        }
        if (component instanceof JTree) {
            JTree tree = (JTree)component; JSONArray nodes = new JSONArray();
            if (tree.getModel().getRoot() != null) treeNodes(tree, tree.getModel().getRoot(), new JSONArray(), nodes);
            value.put("nodes", nodes);
        }
        if (component instanceof JTable) {
            JTable table = (JTable)component; JSONArray rows = new JSONArray();
            for (int row = 0; row < Math.min(table.getRowCount(), 150); row++) {
                JSONArray cells = new JSONArray();
                for (int column = 0; column < Math.min(table.getColumnCount(), 16); column++)
                    cells.add(String.valueOf(table.getValueAt(row, column)));
                rows.add(cells);
            }
            value.put("rows", rows);
        }
        if (component instanceof JList) {
            JList list = (JList)component; JSONArray options = new JSONArray();
            for (int index = 0; index < Math.min(list.getModel().getSize(), 200); index++)
                options.add(String.valueOf(list.getModel().getElementAt(index)));
            value.put("options", options);
            value.put("selected_index", list.getSelectedIndex());
        }
        if (depth > 0 && component instanceof Container) {
            JSONArray children = new JSONArray();
            Component[] source = component instanceof JMenu ? ((JMenu)component).getMenuComponents() : ((Container)component).getComponents();
            for (Component child : source) children.add(snapshot(child, depth - 1));
            value.put("children", children);
        }
        return value;
    }

    /** Executes one registered widget action and reports its result. */
    public static JSONObject perform(JSONObject request) {
        Component component = lookup(request.get("id"));
        if (component == null) throw new IllegalArgumentException("Unknown or expired widget id");
        validateGuard(request, component);
        String action = String.valueOf(request.get("action"));
        switch (action) {
            case "close_dialog":
                if (!(component instanceof Dialog))
                    throw new IllegalArgumentException("Only dialog windows can be closed");
                component.dispatchEvent(new WindowEvent((Dialog)component, WindowEvent.WINDOW_CLOSING));
                break;
            case "focus": {
                JSONObject result = new JSONObject();
                result.put("status", "success");
                result.put("focused", component.requestFocusInWindow());
                return result;
            }
            case "set_number":
                if (!component.isEnabled()) throw new IllegalArgumentException("Number control is disabled");
                if (component.getClass().getName().equals("com.live2d.ui.control.CSlidableFloat$b"))
                    commitFloat(component, ((Number)request.get("value")).floatValue());
                else if (component.getClass().getName().equals("com.live2d.ui.control.CSlidableInt$b")) {
                    try {
                        commitInteger(component, ((Number)request.get("value")).intValue(),
                            Class.forName("com.live2d.ui.control.a.a.i"), Class.forName("com.live2d.ui.control.a.a.m"));
                    } catch (ClassNotFoundException error) { throw new IllegalStateException("Unsupported editor version", error); }
                } else throw new IllegalArgumentException("Widget is not a supported Cubism numeric control");
                break;
            case "capture": {
                int width = component.getWidth(), height = component.getHeight();
                if (width <= 0 || height <= 0 || width > 4096 || height > 4096)
                    throw new IllegalArgumentException("Invalid component capture size");
                double scale = Math.min(2, 2048.0 / Math.max(width, height));
                int imageWidth = Math.max(1, (int)(width * scale));
                int imageHeight = Math.max(1, (int)(height * scale));
                BufferedImage image = new BufferedImage(imageWidth, imageHeight, BufferedImage.TYPE_INT_ARGB);
                Graphics2D graphics = image.createGraphics();
                graphics.scale((double)imageWidth / width, (double)imageHeight / height);
                ArrayList<Component> panels = new ArrayList<>();
                try {
                    preparePrint(component, panels, (double)width / imageWidth, (double)height / imageHeight);
                    component.printAll(graphics);
                    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                    ImageIO.write(image, "png", bytes);
                    JSONObject captured = new JSONObject(); captured.put("status", "success");
                    captured.put("png_base64", Base64.getEncoder().encodeToString(bytes.toByteArray()));
                    // Capture JSON is ASCII; reserve space for the bounded request ID envelope.
                    if (captured.toJSONString().length() > MAX_RESPONSE_BYTES - 128)
                        throw new IllegalArgumentException("Component capture exceeds the transport response limit");
                    return captured;
                } catch (Exception error) { throw new IllegalStateException("Component capture failed", error); }
                finally {
                    graphics.dispose();
                    for (Component panel : panels) try { panel.getClass().getMethod("releasePrint").invoke(panel); } catch (Exception ignored) { }
                }
            }
            case "click":
                if (!(component instanceof AbstractButton)) throw new IllegalArgumentException("Widget is not a button");
                if (!component.isEnabled()) throw new IllegalArgumentException("Button is disabled");
                ((AbstractButton)component).doClick(0); break;
            case "set_text":
                if (!(component instanceof JTextComponent)) throw new IllegalArgumentException("Widget is not a text field");
                if (!((JTextComponent)component).isEditable())
                    throw new IllegalArgumentException("Text field is read-only");
                ((JTextComponent)component).setText(String.valueOf(request.get("value")));
                if (Boolean.TRUE.equals(request.get("enter")) && component instanceof JTextField) {
                    KeyListener[] listeners = component.getKeyListeners();
                    if (listeners.length > 0) {
                        KeyEvent enter = new KeyEvent(component, KeyEvent.KEY_PRESSED, System.currentTimeMillis(), 0, KeyEvent.VK_ENTER, '\n');
                        for (KeyListener listener : listeners) listener.keyPressed(enter);
                    }
                    ((JTextField)component).postActionEvent();
                    FocusEvent blur = new FocusEvent(component, FocusEvent.FOCUS_LOST);
                    for (FocusListener listener : component.getFocusListeners()) listener.focusLost(blur);
                }
                break;
            case "select_combo": {
                if (!(component instanceof JComboBox)) throw new IllegalArgumentException("Widget is not a combo box");
                JComboBox combo = (JComboBox)component;
                int found = request.get("index") instanceof Number ? ((Number)request.get("index")).intValue() : -1;
                if (request.containsKey("value")) {
                    String wanted = String.valueOf(request.get("value")).trim();
                    for (int index = 0; index < combo.getItemCount(); index++) {
                        if (String.valueOf(combo.getItemAt(index)).trim().equals(wanted)) {
                            if (found >= 0) throw new IllegalArgumentException("Ambiguous combo item");
                            found = index;
                        }
                    }
                }
                if (found < 0 || found >= combo.getItemCount()) throw new IllegalArgumentException("Unknown combo item");
                combo.setSelectedIndex(found); break;
            }
            case "select_tree": {
                if (!(component instanceof JTree)) throw new IllegalArgumentException("Widget is not a tree");
                JTree tree = (JTree)component; TreeModel model = tree.getModel();
                ArrayList<TreePath> paths = new ArrayList<>();
                for (Object rawPath : (JSONArray)request.get("paths")) {
                    Object node = model.getRoot(); ArrayList<Object> chain = new ArrayList<>(); chain.add(node);
                    for (Object rawIndex : (JSONArray)rawPath) {
                        int index = ((Number)rawIndex).intValue();
                        if (index < 0 || index >= model.getChildCount(node)) throw new IllegalArgumentException("Tree path no longer exists");
                        node = model.getChild(node, index); chain.add(node);
                    }
                    paths.add(new TreePath(chain.toArray()));
                }
                tree.setSelectionPaths(paths.toArray(new TreePath[0]));
                if (!paths.isEmpty()) tree.scrollPathToVisible(paths.get(0));
                break;
            }
            case "find_table_rows": {
                if (!(component instanceof JTable)) throw new IllegalArgumentException("Widget is not a table");
                JTable table = (JTable)component;
                int column = ((Number)request.get("column")).intValue();
                if (column < 0 || column >= table.getColumnCount()) throw new IllegalArgumentException("Invalid table column");
                JSONArray names = (JSONArray)request.get("values");
                Map<String, Integer> matches = new HashMap<>();
                for (Object name : names) {
                    if (!(name instanceof String) || ((String)name).isEmpty() || matches.containsKey(name))
                        throw new IllegalArgumentException("Table names must be nonempty and unique");
                    matches.put((String)name, -1);
                }
                for (int row = 0; row < table.getRowCount(); row++) {
                    String name = String.valueOf(table.getValueAt(row, column));
                    if (matches.containsKey(name)) {
                        if (matches.get(name) >= 0) throw new IllegalArgumentException("Ambiguous table name: " + name);
                        matches.put(name, row);
                    }
                }
                JSONArray rows = new JSONArray();
                for (Object name : names) {
                    int row = matches.get(name);
                    if (row < 0) throw new IllegalArgumentException("Unknown table name: " + name);
                    rows.add(row);
                }
                JSONObject result = new JSONObject(); result.put("status", "success"); result.put("rows", rows);
                return result;
            }
            case "select_table": {
                if (!(component instanceof JTable)) throw new IllegalArgumentException("Widget is not a table");
                if (!(request.get("rows") instanceof JSONArray)) throw new IllegalArgumentException("Expected table row indices");
                JTable table = (JTable)component;
                JSONArray requested = (JSONArray)request.get("rows");
                int[] rows = new int[requested.size()];
                for (int index = 0; index < requested.size(); index++) {
                    Object raw = requested.get(index);
                    if (!(raw instanceof Number)) throw new IllegalArgumentException("Expected a numeric table row");
                    double row = ((Number)raw).doubleValue();
                    if (!Double.isFinite(row) || row != Math.rint(row) || row < 0 || row >= table.getRowCount())
                        throw new IllegalArgumentException("Table row no longer exists");
                    rows[index] = (int)row;
                }
                table.clearSelection();
                for (int row : rows) {
                    table.addRowSelectionInterval(row, row);
                    table.scrollRectToVisible(table.getCellRect(row, Math.min(2, table.getColumnCount()-1), true));
                }
                break;
            }
            case "scroll_table_row": {
                if (!(component instanceof JTable)) throw new IllegalArgumentException("Widget is not a table");
                JTable table = (JTable)component;
                int row = ((Number)request.get("row")).intValue();
                if (row < 0 || row >= table.getRowCount()) throw new IllegalArgumentException("Table row no longer exists");
                table.scrollRectToVisible(table.getCellRect(row, Math.min(2, table.getColumnCount()-1), true));
                break;
            }
            case "select_list": {
                if (!(component instanceof JList)) throw new IllegalArgumentException("Widget is not a list");
                JList list = (JList)component; int index = ((Number)request.get("index")).intValue();
                if (index < 0 || index >= list.getModel().getSize()) throw new IllegalArgumentException("Invalid list selection");
                list.setSelectedIndex(index); list.ensureIndexIsVisible(index); break;
            }
            case "mouse": {
                int x = ((Number)request.get("x")).intValue();
                int y = ((Number)request.get("y")).intValue();
                int count = request.get("count") instanceof Number ? ((Number)request.get("count")).intValue() : 1;
                int modifiers = request.get("modifiers") instanceof Number ? ((Number)request.get("modifiers")).intValue() : 0;
                component.dispatchEvent(new MouseEvent(component, MouseEvent.MOUSE_MOVED, System.currentTimeMillis(), modifiers, x, y, 0, false, MouseEvent.NOBUTTON));
                for (int event : new int[]{MouseEvent.MOUSE_PRESSED, MouseEvent.MOUSE_RELEASED, MouseEvent.MOUSE_CLICKED})
                    component.dispatchEvent(new MouseEvent(component, event, System.currentTimeMillis(), modifiers, x, y, count, false, MouseEvent.BUTTON1));
                break;
            }
            case "drag": {
                int x = ((Number)request.get("x")).intValue(), y = ((Number)request.get("y")).intValue();
                int endX = ((Number)request.get("to_x")).intValue(), endY = ((Number)request.get("to_y")).intValue();
                int modifiers = request.get("modifiers") instanceof Number ? ((Number)request.get("modifiers")).intValue() : 0;
                component.dispatchEvent(new MouseEvent(component, MouseEvent.MOUSE_MOVED, System.currentTimeMillis(), modifiers, x, y, 0, false, MouseEvent.NOBUTTON));
                component.dispatchEvent(new MouseEvent(component, MouseEvent.MOUSE_PRESSED, System.currentTimeMillis(), modifiers | MouseEvent.BUTTON1_DOWN_MASK, x, y, 1, false, MouseEvent.BUTTON1));
                for (int step = 1; step <= 20; step++) {
                    int nextX = x + (endX-x)*step/20, nextY = y + (endY-y)*step/20;
                    component.dispatchEvent(new MouseEvent(component, MouseEvent.MOUSE_DRAGGED, System.currentTimeMillis(), modifiers | MouseEvent.BUTTON1_DOWN_MASK, nextX, nextY, 0, false, MouseEvent.NOBUTTON));
                }
                component.dispatchEvent(new MouseEvent(component, MouseEvent.MOUSE_RELEASED, System.currentTimeMillis(), modifiers, endX, endY, 1, false, MouseEvent.BUTTON1));
                break;
            }
            case "scroll_into_view":
                if (!(component instanceof JComponent)) throw new IllegalArgumentException("Widget cannot scroll");
                ((JComponent)component).scrollRectToVisible(new Rectangle(0, 0, component.getWidth(), component.getHeight()));
                break;
            default: throw new IllegalArgumentException("Unsupported widget action");
        }
        JSONObject result = new JSONObject(); result.put("status", "success"); return result;
    }

    /** Resolves logical Swing ownership, including detached menu popups. */
    static boolean isOwnedBy(Component component, Component owner) {
        Set<Component> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        while (component != null && seen.add(component)) {
            if (component == owner) return true;
            // A dialog's owner frame is not its widget hierarchy's containing window.
            if (component instanceof Window) return false;
            component = component instanceof JPopupMenu
                ? ((JPopupMenu)component).getInvoker() : component.getParent();
        }
        return false;
    }

    /** Rechecks widget availability, document and modal ownership immediately before acting. */
    private static void validateGuard(JSONObject request, Component component) {
        String action = String.valueOf(request.get("action"));
        if (!component.isEnabled() && !action.equals("capture") && !action.equals("find_table_rows"))
            throw new IllegalArgumentException("Widget is disabled");
        if (!request.containsKey("expected_document")) return;
        String expected = String.valueOf(request.get("expected_document"));
        Frame main = null;
        for (Window window : Window.getWindows()) if (window.isShowing()) {
            if (main == null && window instanceof Frame) main = (Frame)window;
        }
        String title = main == null ? "" : main.getTitle();
        int separator = title.indexOf(" - ");
        if (separator < 0 || !title.substring(separator + 3).equals(expected))
            throw new IllegalArgumentException("The active Cubism document changed");
        Window owner = null;
        for (Window window : Window.getWindows())
            if (window.isShowing() && isOwnedBy(component, window)) { owner = window; break; }
        if (owner == null || !owner.isShowing())
            throw new IllegalArgumentException("Unknown or expired widget");
        if (modalBlocker(owner) != null)
            throw new IllegalArgumentException("Resolve the active modal before editing the document");
    }

    /** Refuses symlinks, shared artifacts, and files owned by a different local user. */
    static void requirePrivate(Path path, boolean directory) throws IOException {
        PosixFileAttributes attributes = Files.readAttributes(path, PosixFileAttributes.class, LinkOption.NOFOLLOW_LINKS);
        UserPrincipal current = FileSystems.getDefault().getUserPrincipalLookupService()
            .lookupPrincipalByName(System.getProperty("user.name"));
        if (attributes.isSymbolicLink() || (directory ? !attributes.isDirectory() : !attributes.isRegularFile())
            || !attributes.owner().equals(current))
            throw new IOException("Unsafe bridge transport artifact");
        for (PosixFilePermission permission : attributes.permissions())
            if (permission.name().startsWith("GROUP_") || permission.name().startsWith("OTHERS_"))
                throw new IOException("Bridge transport must be private to the current user");
    }

    /** Publishes a new private file without following or overwriting a preexisting artifact. */
    static void writePrivate(Path path, String value) throws IOException {
        Files.createFile(path, PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rw-------")));
        Files.writeString(path, value, StandardCharsets.UTF_8, StandardOpenOption.WRITE, LinkOption.NOFOLLOW_LINKS);
    }

    /** Starts a private local request queue inside the already running editor JVM. */
    public static void agentmain(String directory, Instrumentation instrumentation) throws Exception {
        Path supplied = Paths.get(directory).toAbsolutePath();
        requirePrivate(supplied, true);
        Path root = supplied.toRealPath();
        instrumentation.redefineModule(Window.class.getModule(), Set.of(), Map.of(),
            Map.of("java.awt", Set.of(WidgetReference.class.getModule())), Set.of(), Map.of());
        Window.class.getDeclaredMethod("getModalBlocker").setAccessible(true);
        if (!running.compareAndSet(false, true)) throw new IllegalStateException("Swing bridge already running");
        try { writePrivate(root.resolve("ready.json"), "{\"pid\":" + ProcessHandle.current().pid() + "}"); }
        catch (Exception error) { running.set(false); throw error; }
        Thread worker = new Thread(() -> {
            long lastRequest = System.currentTimeMillis();
            while (running.get() && System.currentTimeMillis() - lastRequest < 3600000) {
                Path source = root.resolve("request.json");
                try {
                    if (!Files.exists(source, LinkOption.NOFOLLOW_LINKS)) { Thread.sleep(50); continue; }
                    requirePrivate(source, false);
                    if (Files.size(source) > 1048576) throw new IOException("Bridge request exceeds its size limit");
                    JSONObject request = (JSONObject)new JSONParser().parse(Files.readString(source));
                    String requestId = String.valueOf(request.get("request_id"));
                    if (!requestId.matches("[A-Za-z0-9_-]{1,80}")) throw new IllegalArgumentException("Invalid request id");
                    JSONObject[] response = new JSONObject[1];
                    try {
                        String action = String.valueOf(request.get("action"));
                        if (action.equals("shutdown")) {
                            running.set(false); response[0] = new JSONObject(); response[0].put("status", "stopped");
                        } else if (action.equals("snapshot")) {
                            SwingUtilities.invokeAndWait(() -> {
                                JSONArray windows = new JSONArray();
                                int depth = request.get("depth") instanceof Number ? ((Number)request.get("depth")).intValue() : 36;
                                for (Window window : Window.getWindows()) if (window.isShowing()) windows.add(snapshot(window, Math.min(depth, 48)));
                                response[0] = new JSONObject(); response[0].put("windows", windows); response[0].put("status", "success");
                            });
                        } else if (action.equals("click") || action.equals("mouse") || action.equals("drag") || action.equals("close_dialog")) {
                            SwingUtilities.invokeAndWait(() -> {
                                Object value = request.get("id");
                                Component target = lookup(value);
                                if (target == null) throw new IllegalArgumentException("Unknown or expired widget id");
                                validateGuard(request, target);
                            });
                            SwingUtilities.invokeLater(() -> perform(request));
                            response[0] = new JSONObject(); response[0].put("status", "accepted");
                        } else {
                            SwingUtilities.invokeAndWait(() -> response[0] = perform(request));
                        }
                    } catch (Throwable error) {
                        response[0] = new JSONObject(); response[0].put("status", "error");
                        Throwable cause = error.getCause() == null ? error : error.getCause();
                        response[0].put("error", cause.toString());
                    }
                    response[0].put("request_id", requestId);
                    Path temporary = root.resolve("response-" + requestId + ".tmp");
                    Path destination = root.resolve("response-" + requestId + ".json");
                    if (Files.exists(destination, LinkOption.NOFOLLOW_LINKS)) throw new IOException("Bridge response already exists");
                    writePrivate(temporary, response[0].toJSONString());
                    Files.delete(source);
                    Files.move(temporary, destination, StandardCopyOption.ATOMIC_MOVE);
                    lastRequest = System.currentTimeMillis();
                } catch (Throwable error) {
                    // Stop rather than replay an action whose response could not be published.
                    running.set(false);
                    try { writePrivate(root.resolve("bridge-error.txt"), error.toString()); }
                    catch (Exception ignored) { }
                }
            }
            running.set(false);
            try { Files.deleteIfExists(root.resolve("ready.json")); } catch (Exception ignored) { }
        }, "Live2D-Swing-Bridge");
        worker.setDaemon(true); worker.start();
    }
}
