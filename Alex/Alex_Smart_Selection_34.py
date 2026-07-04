bl_info = {
    "name": "Alex Smart Selection",
    "author": "Alex Khoz",
    "version": (1, 0, 21),
    "blender": (4, 5, 2),
    "location": "View3D > F3 (search) / Keymap",
    "description": "Press B once for waiting-mode (zebra crosshair), B twice for smart box select. Press C once for standard circle select, C twice quickly for smart circle select with similarity matching. Only works in Object mode.",
    "category": "3D View",
}

import bpy
import gpu
import math
import blf
import time
from gpu_extras.batch import batch_for_shader
from bpy.props import IntProperty, BoolProperty, FloatProperty, FloatVectorProperty, EnumProperty
from bpy_extras import view3d_utils

# zebra dash settings (pixels)
DASH_PIXELS = 3
GAP_PIXELS = 0

# box selection dash pattern settings (pixels)
BOX_DASH_LENGTH = 3.0  # Length of each dash segment for box selection
BOX_GAP_LENGTH = 3.0   # Length of gap between dashes for box selection

# circle selection dash pattern settings (pixels)
CIRCLE_DASH_LENGTH = 3.0  # Length of each dash segment
CIRCLE_GAP_LENGTH = 3.0   # Length of gap between dashes

# gradient visualization settings (configurable via Addon Preferences)
# These are now user-configurable in Edit > Preferences > Add-ons > Alex Smart Selection
# Default values (used as fallback if preferences not available):
GRADIENT_STEPS = 50  # Number of ring layers for smooth gradient
GRADIENT_MAX_ALPHA = 0.3  # Maximum opacity (0.0-1.0)
GRADIENT_CURVE_POWER = 0.998  # Growth speed curve (lower=faster, higher=slower)
GRADIENT_MIN_EXTENT_INNER = 1.0  # Min inward extent (%) for values < 100%
GRADIENT_MAX_EXTENT_INNER = 8.0  # Max inward extent (%) for values < 100%
GRADIENT_MIN_EXTENT_OUTER = 1.0  # Min outward extent (%) for values > 100%
GRADIENT_MAX_EXTENT_OUTER = 15.0  # Max outward extent (%) for values > 100%
GRADIENT_MIN_INTENSITY = 0.1  # Minimum alpha intensity (0.0-0.5)
GRADIENT_MAX_CAP_PERCENTAGE = 50000.0  # Max percentage cap for gradient calculation
GRADIENT_EASE = 2.5  # Ease factor for gradient fade (higher = smoother/easier, lower = sharper)

# track the instance of the "waiting" modal and the custom modal
_last_toggle_modal = None
_last_custom_modal = None
_last_circle_toggle_modal = None
_last_circle_custom_modal = None

# Remember last tolerance multiplier step used by user
_last_tolerance_multiplier_step = -1  # -1 = 100% base

# Remember Ctrl toggle state between operator invocations
_global_ctrl_pressed = False
_circle_press_time = 0.0
_circle_double_press_threshold = 0.25  # seconds - reduced for better double-press detection
_circle_pending_timer = None
_circle_should_call_standard = False
_circle_last_radius = None  # Remember last circle radius between uses (None = use preference default)
_circle_use_dashed = True  # Toggle between dashed and solid line style


def _safe_getattr(obj, attr_name, default=None):
    """Safely get an attribute from an object, returning default if ReferenceError occurs."""
    try:
        return getattr(obj, attr_name, default)
    except (ReferenceError, AttributeError):
        return default

def _find_3dview_override():
    """Return a context override for a 3D View area/region, or None."""
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        override = {
                            'window': window,
                            'screen': screen,
                            'area': area,
                            'region': region,
                            'scene': bpy.context.scene,
                            'space_data': area.spaces.active,
                        }
                        return override
    return None


def get_vertex_count(obj, depsgraph, use_evaluated):
    if obj.type != 'MESH':
        return 0
    if use_evaluated:
        eval_obj = obj.evaluated_get(depsgraph)
        try:
            mesh = eval_obj.to_mesh()
            count = len(mesh.vertices)
        except Exception:
            count = len(obj.data.vertices)
        finally:
            try:
                eval_obj.to_mesh_clear()
            except Exception:
                pass
        return count
    else:
        return len(obj.data.vertices)


def get_light_type(obj):
    """Get the light type string for a light object.
    
    Returns: 'POINT', 'SUN', 'SPOT', 'AREA', or None if not a light.
    Uses obj.data.type directly as user suggested.
    """
    if obj.type != 'LIGHT':
        return None
    try:
        # Use obj.data.type directly as user suggested (bpy.data.lights["Light.006"].type)
        light_type = obj.data.type
        
        # Handle string type
        if isinstance(light_type, str):
            # Already a string, return uppercase
            return light_type.upper()
        
        # Handle enum (common in Blender)
        # Try to get the enum value as string
        if hasattr(light_type, 'name'):
            return light_type.name.upper()
        
        # Try to convert to string and uppercase
        if light_type is not None:
            type_str = str(light_type)
            # Remove any prefix like 'LIGHT_' or similar
            if '_' in type_str:
                type_str = type_str.split('_')[-1]
            return type_str.upper()
        
        return None
    except (AttributeError, TypeError, ValueError):
        return None


def get_text_content(obj):
    """Extract text content from a text (FONT) object.
    
    Returns:
        str: Text content or None if not a text object or error.
    """
    try:
        if obj.type != 'FONT':
            return None
        data = getattr(obj, 'data', None)
        if data is None:
            return None
        # Get text body - it's stored in body attribute
        text_body = getattr(data, 'body', '')
        return text_body
    except (AttributeError, TypeError, ValueError):
        return None


def get_object_metric(obj, depsgraph, use_evaluated):
    """Return a (type, numeric metric) tuple for similarity comparisons.

    Meshes -> vertex count.
    Empties -> size*100 + child count.
    Curves/Surfaces/Fonts -> total control points across splines.
    Armatures -> bone count.
    Lights -> energy (power) value.
    Cameras -> sensor width.
    Others -> 0.0
    """
    t = obj.type
    metric = 0.0

    if t == 'MESH':
        metric = float(get_vertex_count(obj, depsgraph, use_evaluated))
    elif t == 'EMPTY':
        size = getattr(obj, 'empty_display_size', 0.0)
        metric = float(int(size * 100.0) + len(obj.children))
    elif t in {'CURVE', 'SURFACE', 'FONT', 'META'}:
        tot = 0
        data = getattr(obj, 'data', None)
        if data is not None:
            for s in getattr(data, 'splines', []):
                if hasattr(s, 'bezier_points'):
                    tot += len(s.bezier_points)
                elif hasattr(s, 'points'):
                    tot += len(s.points)
        metric = float(tot)
    elif t == 'ARMATURE':
        data = getattr(obj, 'data', None)
        if data is not None:
            metric = float(len(data.bones))
    elif t == 'LIGHT':
        data = getattr(obj, 'data', None)
        # Use energy (power) as the metric for lights
        metric = float(getattr(data, 'energy', 1.0)) if data is not None else 1.0
    elif t == 'CAMERA':
        data = getattr(obj, 'data', None)
        metric = float(getattr(data, 'sensor_width', 0.0)) if data is not None else 0.0
    else:
        metric = 0.0

    return t, metric


def _get_tolerance_multiplier_steps():
    """Return list of tolerance multiplier increments with fine gradations.
    
    Pattern: Fine steps near 100%, gradually increasing:
    - First step: 10 (110%)
    - Fine gradations: 5, 5, 5, 5, 5 (115, 120, 125, 130, 135%)
    - Medium gradations: 10, 10, 10, 10, 10, 10 (145, 155, 165, 175, 185, 195%)
    - Gradually increasing: 15, 20, 25, 30... (210, 230, 255, 285...)
    - Larger increments: 50, 60, 70... for very high values
    """
    steps = []
    
    # Step 1: First step to 110% (10% increment)
    steps.append(10)
    
    # Step 2: Fine gradations near 110% (5% increments): 115, 120, 125, 130, 135
    for i in range(5):
        steps.append(5)
    
    # Step 3: Medium gradations (10% increments): 145, 155, 165, 175, 185, 195
    for i in range(6):
        steps.append(10)
    
    # Step 4: Gradually increasing increments (15, 20, 25, 30, ...)
    increment = 15
    for i in range(10):
        steps.append(increment)
        increment += 5
    
    # Step 5: Continue with larger increments (50, 60, 70, 80, ...)
    increment = 50
    for i in range(15):
        steps.append(increment)
        increment += 10
    
    # Step 6: Even larger increments for very high values (200, 250, 300, ..., 1200)
    increment = 200
    for i in range(20):
        steps.append(increment)
        increment += 50
    
    # Step 7: Continue with large increments (1250, 1500, 1750, ..., 5000)
    increment = 1250
    for i in range(15):
        steps.append(increment)
        increment += 250
    
    # Step 8: Very large increments (5500, 6000, 6500, ..., 15000)
    increment = 5500
    for i in range(19):
        steps.append(increment)
        increment += 500
    
    # Step 9: Massive increments (16000, 17000, 18000, ..., 50000)
    increment = 16000
    for i in range(34):
        steps.append(increment)
        increment += 1000
    
    # Step 10: Extreme increments (55000, 60000, 65000, ..., 150000)
    increment = 55000
    for i in range(19):
        steps.append(increment)
        increment += 5000
    
    # Step 11: Maximum increments (200000, 250000, 300000, ..., 500000)
    increment = 200000
    for i in range(10):
        steps.append(increment)
        increment += 50000
    
    return steps

def _get_decreasing_multiplier_values():
    """Generate decreasing multiplier values from 90 to 0.01 with fine gradations.
    
    Returns a list: [90, 80, 70, ..., 25, 20, 15, 14, 13, ..., 5, 4.5, 4.0, ..., 2.1, 2.0, 1.9, ..., 1.0, 0.99, 0.98, ...]
    Note: 100% is handled separately as step -1 (base)
    """
    values = []
    
    # Step 1: 90 down to 30 in steps of 10
    for v in range(90, 29, -10):
        values.append(float(v))
    
    # Step 2: 25, 20, 15
    values.extend([25.0, 20.0, 15.0])
    
    # Step 3: 14 down to 5 in steps of 1
    for v in range(14, 4, -1):
        values.append(float(v))
    
    # Step 4: 4.5 down to 2.5 in steps of 0.5
    for v in range(45, 24, -5):  # Using integer math: 45 = 4.5*10, 24 = 2.4*10
        values.append(v / 10.0)
    
    # Step 5: 2.4 down to 2.0 in steps of 0.1
    for v in range(24, 19, -1):  # 24 = 2.4*10, 19 = 1.9*10
        values.append(v / 10.0)
    
    # Step 6: 1.9 down to 1.1 in steps of 0.1
    for v in range(19, 10, -1):  # 19 = 1.9*10, 10 = 1.0*10
        values.append(v / 10.0)
    
    # Step 7: 1.0, then 0.99 down to 0.01 in steps of 0.01
    values.append(1.0)
    for v in range(99, 0, -1):  # 99 = 0.99*100, down to 1 = 0.01*100
        values.append(v / 100.0)
    
    return values

def _multiplier_to_step_index(multiplier, increasing=True):
    """Convert a multiplier value to a step index."""
    if increasing:
        steps = _get_tolerance_multiplier_steps()
        if multiplier <= 100.0:
            return -1
        step = -1
        total = 100.0
        for i, s in enumerate(steps):
            total += s
            step = i
            if total >= multiplier:
                return step
        return step
    else:
        decreasing_values = _get_decreasing_multiplier_values()
        try:
            # Find closest index
            for i, val in enumerate(decreasing_values):
                if val <= multiplier:
                    return -(i + 1)
            return -(len(decreasing_values) + 1)
        except:
            return -1

def _step_index_to_multiplier(step_index):
    """Convert a step index to a multiplier value."""
    if step_index == -1:
        # Base value: 100%
        return 100.0
    elif step_index >= 0:
        # Increasing: use steps
        steps = _get_tolerance_multiplier_steps()
        multiplier = 100.0
        for i in range(min(step_index + 1, len(steps))):
            multiplier += steps[i]
        return multiplier
    else:
        # Decreasing: use decreasing values
        # step -2 = index 0 (90), step -3 = index 1 (80), etc.
        decreasing_values = _get_decreasing_multiplier_values()
        abs_index = abs(step_index) - 2  # -2 because step -1 is 100%, step -2 is first decreasing value
        if 0 <= abs_index < len(decreasing_values):
            return decreasing_values[abs_index]
        else:
            return 0.01  # Minimum value

def _get_addon_preferences():
    """Get addon preferences with fallback to defaults."""
    try:
        if hasattr(bpy.context, 'preferences'):
            addon_name = bl_info.get("name", "")
            if addon_name and addon_name in bpy.context.preferences.addons:
                return bpy.context.preferences.addons[addon_name].preferences
            addon_key = __package__ if __package__ else __name__.split('.')[0]
            if addon_key in bpy.context.preferences.addons:
                return bpy.context.preferences.addons[addon_key].preferences
    except (KeyError, AttributeError):
        pass
    return None

def _should_disable_continuous_selection(context, calculation_time_ms=None):
    """
    Check if continuous selection should be disabled based on preferences and scene state.
    
    Args:
        context: Blender context
        calculation_time_ms: Optional calculation time in milliseconds from previous frame
    
    Returns:
        tuple: (should_disable, reason) - True if continuous selection should be disabled
    """
    prefs = _get_addon_preferences()
    if not prefs or not prefs.auto_disable_continuous_selection:
        return (False, None)
    
    # Check object count
    object_count = len(context.view_layer.objects)
    if object_count > prefs.max_objects_for_continuous:
        return (True, f"Too many objects ({object_count} > {prefs.max_objects_for_continuous})")
    
    # Check calculation time if provided
    if calculation_time_ms is not None and calculation_time_ms > prefs.max_time_for_continuous:
        return (True, f"Calculation too slow ({calculation_time_ms:.1f}ms > {prefs.max_time_for_continuous}ms)")
    
    return (False, None)

def _format_multiplier_display(multiplier):
    """Format multiplier for display, showing appropriate decimal places."""
    if multiplier >= 10.0:
        # For values >= 10, show as integer
        return f"{int(multiplier)}%"
    elif multiplier >= 1.0:
        # For values 1.0 to 9.9, show 1 decimal place if needed
        if multiplier == int(multiplier):
            return f"{int(multiplier)}%"
        else:
            return f"{multiplier:.1f}%"
    else:
        # For values < 1.0, show 2 decimal places
        return f"{multiplier:.2f}%"


def _get_vertex_count_threshold(operator_instance):
    """Get the count threshold based on reference object and tolerance multiplier.
    
    Returns the threshold count that represents the selection range boundary:
    - For meshes: vertex count
    - For curves/surfaces/fonts: control point count
    - For armatures: bone count
    - For lights: power (energy) value (float)
    - For other types: metric value
    
    - For multiplier < 100%: Returns the minimum threshold (lower bound)
    - For multiplier == 100%: Returns the reference metric
    - For multiplier > 100%: Returns the maximum threshold (upper bound)
    
    Returns 0 if operator_instance or reference metrics are not available.
    Returns a tuple (value, type) where type is 'LIGHT' for lights, None otherwise.
    """
    try:
        ref_metrics = getattr(operator_instance, '_ref_metrics', [])
        tolerance_multiplier = getattr(operator_instance, 'tolerance_multiplier', 100.0)
        
        if not ref_metrics:
            return (0, None)
        
        # Get the first reference metric (primary reference object)
        # The metric represents:
        # - MESH: vertex count
        # - CURVE/SURFACE/FONT/META: control point count
        # - ARMATURE: bone count
        # - LIGHT: power (energy) value
        # - Others: appropriate metric value
        ref_type, ref_metric = ref_metrics[0]
        
        if tolerance_multiplier == 100.0:
            # At 100%, return the reference metric itself
            if ref_type == 'LIGHT':
                return (ref_metric, 'LIGHT')  # Return float for lights
            else:
                return (int(round(ref_metric)), None)
        
        # Calculate threshold: ref_metric * (tolerance_multiplier / 100.0)
        multiplier_factor = tolerance_multiplier / 100.0
        threshold = ref_metric * multiplier_factor
        
        # For lights, return float; for others, round to integer
        if ref_type == 'LIGHT':
            return (threshold, 'LIGHT')
        else:
            return (int(round(threshold)), None)
    
    except (AttributeError, ReferenceError, TypeError):
        return (0, None)


def _format_power_display(power):
    """Format light power (energy) for display."""
    if power >= 1000.0:
        return f"{power / 1000.0:.2f}kW"
    elif power >= 1.0:
        return f"{power:.2f}W"
    else:
        return f"{power:.3f}W"


def _format_vertex_count_display(vertex_count):
    """Format vertex count for display."""
    if vertex_count >= 1000000:
        return f"{vertex_count / 1000000.0:.1f}M"
    elif vertex_count >= 1000:
        return f"{vertex_count / 1000.0:.1f}K"
    else:
        return str(vertex_count)


def _has_multiple_object_types(operator_instance):
    """Check if multiple different object types are selected as reference objects.
    
    Returns True if there are 2+ different object types in the reference metrics.
    When multiple types are selected (e.g., mesh + light + curve), we should
    always show percentage instead of vertex count or power.
    """
    try:
        ref_metrics = getattr(operator_instance, '_ref_metrics', [])
        if not ref_metrics or len(ref_metrics) < 2:
            return False
        
        # Count unique types
        types = {ref_type for ref_type, _ in ref_metrics}
        return len(types) > 1
    except (AttributeError, ReferenceError, TypeError):
        return False

def _draw_gradient_ring(cx, cy, radius, tolerance_multiplier, num_segments=256):
    """Draw a gradient ring to visualize tolerance multiplier adjustment.
    The gradient starts from the circle edge itself, not from the center.
    Gradient size and intensity increase smoothly as tolerance deviates from 100%.
    
    Args:
        cx, cy: Center coordinates
        radius: Base radius of the circle (the edge where gradient starts)
        tolerance_multiplier: Current tolerance multiplier (100.0 = 100%)
        num_segments: Number of segments for smooth circles
    """
    if tolerance_multiplier == 100.0:
        return  # No gradient at 100%
    
    # Get addon preferences for gradient settings
    addon_prefs = None
    try:
        if hasattr(bpy.context, 'preferences'):
            # Try different methods to find the addon key
            # Method 1: Use bl_info name (most reliable)
            addon_name = bl_info.get("name", "")
            if addon_name and addon_name in bpy.context.preferences.addons:
                addon_prefs = bpy.context.preferences.addons[addon_name].preferences
            # Method 2: Try package/module name
            elif not addon_prefs:
                addon_key = __package__ if __package__ else __name__.split('.')[0]
                if addon_key in bpy.context.preferences.addons:
                    addon_prefs = bpy.context.preferences.addons[addon_key].preferences
    except (KeyError, AttributeError):
        # Fallback to defaults if preferences not available
        pass
    
    # Gradient parameters from preferences or defaults
    if addon_prefs:
        gradient_steps = addon_prefs.gradient_steps
        max_gradient_alpha = addon_prefs.gradient_max_alpha
        gradient_curve_power = addon_prefs.gradient_curve_power
        min_extent_inner = addon_prefs.gradient_min_extent_inner
        max_extent_inner = addon_prefs.gradient_max_extent_inner
        min_extent_outer = addon_prefs.gradient_min_extent_outer
        max_extent_outer = addon_prefs.gradient_max_extent_outer
        min_intensity = addon_prefs.gradient_min_intensity
        max_cap_percentage = addon_prefs.gradient_max_cap_percentage
        gradient_ease = addon_prefs.gradient_ease
    else:
        # Default values from constants (matching preferences defaults)
        gradient_steps = GRADIENT_STEPS
        max_gradient_alpha = GRADIENT_MAX_ALPHA
        gradient_curve_power = GRADIENT_CURVE_POWER
        min_extent_inner = GRADIENT_MIN_EXTENT_INNER
        max_extent_inner = GRADIENT_MAX_EXTENT_INNER
        min_extent_outer = GRADIENT_MIN_EXTENT_OUTER
        max_extent_outer = GRADIENT_MAX_EXTENT_OUTER
        min_intensity = GRADIENT_MIN_INTENSITY
        max_cap_percentage = GRADIENT_MAX_CAP_PERCENTAGE
        gradient_ease = GRADIENT_EASE
        gradient_color = (1.0, 1.0, 1.0, 1.0)  # Default white (RGBA)
    
    # Get gradient color from preferences
    if addon_prefs:
        gradient_color = tuple(addon_prefs.gradient_color)
        # Ensure 4 components (RGBA) - handle old 3-component colors
        if len(gradient_color) == 3:
            gradient_color = gradient_color + (1.0,)
    else:
        gradient_color = (1.0, 1.0, 1.0, 1.0)  # Default white (RGBA)
    
    try:
        gpu.state.blend_set('ALPHA')
    except Exception:
        pass
    
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    if tolerance_multiplier < 100.0:
        # Gradient from circle edge towards center
        # Calculate normalized deviation from 100% (0.0 at 100%, approaching 1.0 as multiplier approaches 0)
        # Use smooth mapping that's responsive to all levels
        normalized_value = tolerance_multiplier / 100.0  # 0.0 to 1.0 (1.0 = 100%, 0.0 = 0%)
        # Invert and apply smooth curve: more deviation = larger gradient
        # Use configurable power curve for better responsiveness at small deviations while still scaling smoothly
        raw_deviation = 1.0 - normalized_value  # 0.0 at 100%, 1.0 at 0%
        deviation_factor = math.pow(raw_deviation, gradient_curve_power)  # Configurable curve power
        
        # Calculate gradient extent (how far inward) - increases smoothly with deviation
        # Minimum extent at small deviations, maximum extent at large deviations
        # Even small deviations should show some visible gradient
        min_extent_ratio = min_extent_inner / 100.0  # Convert percentage to ratio
        max_extent_ratio = max_extent_inner / 100.0  # Convert percentage to ratio
        extent_ratio = min_extent_ratio + (max_extent_ratio - min_extent_ratio) * deviation_factor
        
        # Calculate gradient intensity (alpha) - increases/decreases with deviation for transparency variation
        # Start at minimum intensity for visibility even at small deviations
        # Transparency scales with deviation: more deviation = more opaque, less deviation = more transparent
        intensity_range = 1.0 - min_intensity
        gradient_strength = min_intensity + deviation_factor * intensity_range
        
        # Scale max alpha based on deviation factor to make transparency variation more noticeable
        # This makes the gradient more transparent near 100% and more opaque as deviation increases
        scaled_max_alpha = max_gradient_alpha * (min_intensity + (1.0 - min_intensity) * deviation_factor)
        
        # Draw rings from the circle edge inward
        inner_radius_limit = radius * (1.0 - extent_ratio)  # Limit based on extent
        for i in range(gradient_steps):
            t = i / gradient_steps
            # Outer radius starts at the circle edge (radius)
            outer_radius = radius - (radius - inner_radius_limit) * t
            # Inner radius is one step closer to center
            inner_radius = radius - (radius - inner_radius_limit) * (t + 1.0 / gradient_steps)
            
            # Alpha decreases smoothly as we go inward (fade out)
            # Use eased falloff curve for smoother visual appearance
            # Higher ease value = smoother/easier fade, lower = sharper fade
            fade_curve = math.pow(1.0 - t, gradient_ease)  # Eased falloff
            alpha = scaled_max_alpha * gradient_strength * fade_curve
            
            if alpha <= 0.001 or inner_radius <= 0:
                break
            
            # Create ring vertices using TRI_STRIP (draw between two circles)
            ring_verts = []
            for j in range(num_segments + 1):
                angle = (j / num_segments) * 2 * math.pi
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                # Outer circle vertex (at the edge or closer to edge)
                ring_verts.append((cx + outer_radius * cos_a, cy + outer_radius * sin_a))
                # Inner circle vertex (closer to center)
                ring_verts.append((cx + inner_radius * cos_a, cy + inner_radius * sin_a))
            
            # Draw ring using TRI_STRIP
            batch = batch_for_shader(shader, 'TRI_STRIP', {"pos": ring_verts})
            shader.bind()
            # Multiply calculated alpha by color alpha (if available)
            color_alpha = gradient_color[3] if len(gradient_color) > 3 else 1.0
            final_alpha = alpha * color_alpha
            shader.uniform_float("color", (gradient_color[0], gradient_color[1], gradient_color[2], final_alpha))
            batch.draw(shader)
    
    else:
        # Gradient from circle edge towards outside
        # Calculate normalized deviation from 100% (0.0 at 100%, increasing as multiplier increases)
        # Use logarithmic-like mapping that's responsive to all levels, even with very high caps
        normalized_value = tolerance_multiplier / 100.0  # 1.0 at 100%, 2.0 at 200%, etc.
        
        # Cap value for calculation
        max_cap_normalized = max_cap_percentage / 100.0
        capped_value = min(normalized_value, max_cap_normalized)
        
        # Calculate deviation from 100%
        deviation_raw = capped_value - 1.0  # 0.0 at 100%
        
        if deviation_raw <= 0.0:
            deviation_factor = 0.0
        else:
            # Use logarithmic scaling for better responsiveness across wide ranges
            # This ensures small deviations (like 101%) still show visible gradients
            # while large deviations (like 4000%) don't dominate
            # Apply logarithmic scaling, then normalize and apply power curve
            # Use log base that scales nicely (log(deviation + 1) / log(max_deviation + 1))
            max_deviation = max_cap_normalized - 1.0  # Maximum deviation from 100%
            
            # Logarithmic normalization: maps small and large deviations more evenly
            log_deviation = math.log(deviation_raw + 1.0)
            log_max_deviation = math.log(max_deviation + 1.0)
            normalized_deviation = min(1.0, log_deviation / log_max_deviation) if log_max_deviation > 0 else 0.0
            
            # Apply power curve for fine-tuning
            deviation_factor = math.pow(normalized_deviation, gradient_curve_power) if normalized_deviation > 0 else 0.0
        
        # Calculate gradient extent (how far outward) - increases smoothly with deviation
        # Minimum extent at small deviations, maximum extent at large deviations
        # Even small deviations should show some visible gradient
        min_extent_ratio = min_extent_outer / 100.0  # Convert percentage to ratio
        max_extent_ratio = max_extent_outer / 100.0  # Convert percentage to ratio
        extent_ratio = min_extent_ratio + (max_extent_ratio - min_extent_ratio) * deviation_factor
        
        # Calculate gradient intensity (alpha) - increases/decreases with deviation for transparency variation
        # Start at minimum intensity for visibility even at small deviations
        # Transparency scales with deviation: more deviation = more opaque, less deviation = more transparent
        intensity_range = 1.0 - min_intensity
        gradient_strength = min_intensity + deviation_factor * intensity_range
        
        # Scale max alpha based on deviation factor to make transparency variation more noticeable
        # This makes the gradient more transparent near 100% and more opaque as deviation increases
        scaled_max_alpha = max_gradient_alpha * (min_intensity + (1.0 - min_intensity) * deviation_factor)
        
        # Draw rings from the circle edge outward
        max_outer_radius = radius * (1.0 + extent_ratio)  # Limit based on extent
        for i in range(gradient_steps):
            t = i / gradient_steps
            # Inner radius starts at the circle edge (radius)
            inner_radius = radius + (max_outer_radius - radius) * t
            # Outer radius is one step further out
            outer_radius = radius + (max_outer_radius - radius) * (t + 1.0 / gradient_steps)
            
            # Alpha decreases smoothly as we go outward (fade out)
            # Use eased falloff curve for smoother visual appearance
            # Higher ease value = smoother/easier fade, lower = sharper fade
            fade_curve = math.pow(1.0 - t, gradient_ease)  # Eased falloff
            alpha = scaled_max_alpha * gradient_strength * fade_curve
            
            if alpha <= 0.001:
                break
            
            # Create ring vertices using TRI_STRIP (draw between two circles)
            ring_verts = []
            for j in range(num_segments + 1):
                angle = (j / num_segments) * 2 * math.pi
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                # Inner circle vertex (at the edge or further out)
                ring_verts.append((cx + inner_radius * cos_a, cy + inner_radius * sin_a))
                # Outer circle vertex (further out)
                ring_verts.append((cx + outer_radius * cos_a, cy + outer_radius * sin_a))
            
            # Draw ring using TRI_STRIP
            batch = batch_for_shader(shader, 'TRI_STRIP', {"pos": ring_verts})
            shader.bind()
            # Multiply calculated alpha by color alpha (if available)
            color_alpha = gradient_color[3] if len(gradient_color) > 3 else 1.0
            final_alpha = alpha * color_alpha
            shader.uniform_float("color", (gradient_color[0], gradient_color[1], gradient_color[2], final_alpha))
            batch.draw(shader)
    
    try:
        gpu.state.blend_set('NONE')
    except Exception:
        pass

def _draw_gradient_box(xmin, ymin, xmax, ymax, tolerance_multiplier):
    """Draw a gradient box to visualize tolerance multiplier adjustment.
    The gradient starts from the box edges themselves.
    Gradient size and intensity increase smoothly as tolerance deviates from 100%.
    
    Args:
        xmin, ymin, xmax, ymax: Box coordinates
        tolerance_multiplier: Current tolerance multiplier (100.0 = 100%)
    """
    if tolerance_multiplier == 100.0:
        return  # No gradient at 100%
    
    # Get addon preferences for gradient settings
    addon_prefs = None
    try:
        if hasattr(bpy.context, 'preferences'):
            # Try different methods to find the addon key
            # Method 1: Use bl_info name (most reliable)
            addon_name = bl_info.get("name", "")
            if addon_name and addon_name in bpy.context.preferences.addons:
                addon_prefs = bpy.context.preferences.addons[addon_name].preferences
            # Method 2: Try package/module name
            elif not addon_prefs:
                addon_key = __package__ if __package__ else __name__.split('.')[0]
                if addon_key in bpy.context.preferences.addons:
                    addon_prefs = bpy.context.preferences.addons[addon_key].preferences
    except (KeyError, AttributeError):
        # Fallback to defaults if preferences not available
        pass
    
    # Gradient parameters from preferences or defaults
    if addon_prefs:
        gradient_steps = addon_prefs.gradient_steps
        max_gradient_alpha = addon_prefs.gradient_max_alpha
        gradient_curve_power = addon_prefs.gradient_curve_power
        min_extent_inner = addon_prefs.gradient_min_extent_inner
        max_extent_inner = addon_prefs.gradient_max_extent_inner
        min_extent_outer = addon_prefs.gradient_min_extent_outer
        max_extent_outer = addon_prefs.gradient_max_extent_outer
        min_intensity = addon_prefs.gradient_min_intensity
        max_cap_percentage = addon_prefs.gradient_max_cap_percentage
        gradient_ease = addon_prefs.gradient_ease
    else:
        # Default values from constants (matching preferences defaults)
        gradient_steps = GRADIENT_STEPS
        max_gradient_alpha = GRADIENT_MAX_ALPHA
        gradient_curve_power = GRADIENT_CURVE_POWER
        min_extent_inner = GRADIENT_MIN_EXTENT_INNER
        max_extent_inner = GRADIENT_MAX_EXTENT_INNER
        min_extent_outer = GRADIENT_MIN_EXTENT_OUTER
        max_extent_outer = GRADIENT_MAX_EXTENT_OUTER
        min_intensity = GRADIENT_MIN_INTENSITY
        max_cap_percentage = GRADIENT_MAX_CAP_PERCENTAGE
        gradient_ease = GRADIENT_EASE
        gradient_color = (1.0, 1.0, 1.0, 1.0)  # Default white (RGBA)
    
    # Get gradient color from preferences
    if addon_prefs:
        gradient_color = tuple(addon_prefs.gradient_color)
        # Ensure 4 components (RGBA) - handle old 3-component colors
        if len(gradient_color) == 3:
            gradient_color = gradient_color + (1.0,)
    else:
        gradient_color = (1.0, 1.0, 1.0, 1.0)  # Default white (RGBA)
    
    try:
        gpu.state.blend_set('ALPHA')
    except Exception:
        pass
    
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    # Calculate box center and dimensions
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    # Use smaller dimension for percentage calculation to ensure square-ish gradients
    base_size = min(width, height)
    
    if tolerance_multiplier < 100.0:
        # Gradient from box edge towards center
        normalized_value = tolerance_multiplier / 100.0
        raw_deviation = 1.0 - normalized_value
        deviation_factor = math.pow(raw_deviation, gradient_curve_power)
        
        # Calculate gradient extent (how far inward)
        min_extent_ratio = min_extent_inner / 100.0
        max_extent_ratio = max_extent_inner / 100.0
        extent_ratio = min_extent_ratio + (max_extent_ratio - min_extent_ratio) * deviation_factor
        
        # Calculate gradient intensity
        intensity_range = 1.0 - min_intensity
        gradient_strength = min_intensity + deviation_factor * intensity_range
        scaled_max_alpha = max_gradient_alpha * (min_intensity + (1.0 - min_intensity) * deviation_factor)
        
        # Calculate how far inward to extend
        extent_distance = base_size * extent_ratio / 2.0  # Divide by 2 because we extend from both sides
        
        # Draw gradient layers from edge inward
        for i in range(gradient_steps):
            t = i / gradient_steps
            # Calculate current layer extent from edge (0 at edge, extent_distance at center)
            current_extent = extent_distance * t
            
            # Create outer box for this layer (starts at edge, moves inward)
            outer_xmin = xmin + current_extent
            outer_ymin = ymin + current_extent
            outer_xmax = xmax - current_extent
            outer_ymax = ymax - current_extent
            
            if outer_xmin >= outer_xmax or outer_ymin >= outer_ymax:
                break
            
            # Create inner box (next layer inward)
            if i + 1 < gradient_steps:
                next_extent = extent_distance * (i + 1) / gradient_steps
                inner_xmin = xmin + next_extent
                inner_ymin = ymin + next_extent
                inner_xmax = xmax - next_extent
                inner_ymax = ymax - next_extent
            else:
                # Last layer: inner box is center point (very small)
                center_x = (xmin + xmax) / 2.0
                center_y = (ymin + ymax) / 2.0
                inner_xmin = inner_xmax = center_x
                inner_ymin = inner_ymax = center_y
            
            # Skip if inner is same as or outside outer (for inner gradient, inner should be INSIDE outer)
            # Inner box has larger xmin/ymin and smaller xmax/ymax (it's further inward)
            if inner_xmin <= outer_xmin or inner_ymin <= outer_ymin or inner_xmax >= outer_xmax or inner_ymax >= outer_ymax:
                if i < gradient_steps - 1:
                    continue
            
            # Calculate alpha with eased falloff (strongest at edge t=0, fades toward center)
            fade_curve = math.pow(1.0 - t, gradient_ease)
            alpha = scaled_max_alpha * gradient_strength * fade_curve
            
            if alpha <= 0.001:
                break
            
            # Draw gradient rectangle ring (between outer and inner boxes)
            # Draw as four rectangles (top, bottom, left, right edges) plus four corners to fill gaps
            # Edges exclude corners to avoid overlap, corners fill the gaps seamlessly
            rects = [
                # Top edge (excludes corners)
                [(inner_xmin, outer_ymax), (inner_xmax, outer_ymax), (inner_xmax, inner_ymax), (inner_xmin, inner_ymax)],
                # Bottom edge (excludes corners)
                [(inner_xmin, outer_ymin), (inner_xmin, inner_ymin), (inner_xmax, inner_ymin), (inner_xmax, outer_ymin)],
                # Left edge (excludes corners)
                [(outer_xmin, inner_ymin), (outer_xmin, inner_ymax), (inner_xmin, inner_ymax), (inner_xmin, inner_ymin)],
                # Right edge (excludes corners)
                [(inner_xmax, inner_ymin), (inner_xmax, inner_ymax), (outer_xmax, inner_ymax), (outer_xmax, inner_ymin)],
                # Top-left corner
                [(outer_xmin, outer_ymax), (inner_xmin, outer_ymax), (inner_xmin, inner_ymax), (outer_xmin, inner_ymax)],
                # Top-right corner
                [(inner_xmax, outer_ymax), (outer_xmax, outer_ymax), (outer_xmax, inner_ymax), (inner_xmax, inner_ymax)],
                # Bottom-left corner
                [(outer_xmin, outer_ymin), (outer_xmin, inner_ymin), (inner_xmin, inner_ymin), (inner_xmin, outer_ymin)],
                # Bottom-right corner
                [(inner_xmax, outer_ymin), (inner_xmax, inner_ymin), (outer_xmax, inner_ymin), (outer_xmax, outer_ymin)],
            ]
            
            for rect_verts in rects:
                if len(rect_verts) >= 3:
                    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": rect_verts})
                    shader.bind()
                    # Multiply calculated alpha by color alpha (if available)
                    color_alpha = gradient_color[3] if len(gradient_color) > 3 else 1.0
                    final_alpha = alpha * color_alpha
                    shader.uniform_float("color", (gradient_color[0], gradient_color[1], gradient_color[2], final_alpha))
                    batch.draw(shader)
    
    else:
        # Gradient from box edge towards outside
        normalized_value = tolerance_multiplier / 100.0
        max_cap_normalized = max_cap_percentage / 100.0
        capped_value = min(normalized_value, max_cap_normalized)
        deviation_raw = capped_value - 1.0
        
        if deviation_raw <= 0.0:
            deviation_factor = 0.0
        else:
            max_deviation = max_cap_normalized - 1.0
            log_deviation = math.log(deviation_raw + 1.0)
            log_max_deviation = math.log(max_deviation + 1.0)
            normalized_deviation = min(1.0, log_deviation / log_max_deviation) if log_max_deviation > 0 else 0.0
            deviation_factor = math.pow(normalized_deviation, gradient_curve_power) if normalized_deviation > 0 else 0.0
        
        # Calculate gradient extent (how far outward)
        min_extent_ratio = min_extent_outer / 100.0
        max_extent_ratio = max_extent_outer / 100.0
        extent_ratio = min_extent_ratio + (max_extent_ratio - min_extent_ratio) * deviation_factor
        
        # Calculate gradient intensity
        intensity_range = 1.0 - min_intensity
        gradient_strength = min_intensity + deviation_factor * intensity_range
        scaled_max_alpha = max_gradient_alpha * (min_intensity + (1.0 - min_intensity) * deviation_factor)
        
        # Calculate how far outward to extend
        extent_distance = base_size * extent_ratio / 2.0
        
        # Draw gradient layers from edge outward
        for i in range(gradient_steps):
            t = i / gradient_steps
            # Calculate current layer extent
            current_extent = extent_distance * t
            
            # Create outer box for this layer
            outer_xmin = xmin - current_extent
            outer_ymin = ymin - current_extent
            outer_xmax = xmax + current_extent
            outer_ymax = ymax + current_extent
            
            # Create inner box (previous layer, or original box for first layer)
            if i == 0:
                inner_xmin, inner_ymin, inner_xmax, inner_ymax = xmin, ymin, xmax, ymax
            else:
                prev_extent = extent_distance * (i - 1) / gradient_steps
                inner_xmin = xmin - prev_extent
                inner_ymin = ymin - prev_extent
                inner_xmax = xmax + prev_extent
                inner_ymax = ymax + prev_extent
            
            # Calculate alpha with eased falloff
            fade_curve = math.pow(1.0 - t, gradient_ease)
            alpha = scaled_max_alpha * gradient_strength * fade_curve
            
            if alpha <= 0.001:
                break
            
            # Draw gradient rectangle ring (between inner and outer boxes)
            # Draw as four rectangles (top, bottom, left, right edges) plus four corners to fill gaps
            # Edges exclude corners to avoid overlap, corners fill the gaps seamlessly
            rects = [
                # Top edge (excludes corners)
                [(inner_xmin, inner_ymax), (inner_xmax, inner_ymax), (inner_xmax, outer_ymax), (inner_xmin, outer_ymax)],
                # Bottom edge (excludes corners)
                [(inner_xmin, inner_ymin), (inner_xmin, outer_ymin), (inner_xmax, outer_ymin), (inner_xmax, inner_ymin)],
                # Left edge (excludes corners)
                [(outer_xmin, inner_ymin), (outer_xmin, inner_ymax), (inner_xmin, inner_ymax), (inner_xmin, inner_ymin)],
                # Right edge (excludes corners)
                [(inner_xmax, inner_ymin), (inner_xmax, inner_ymax), (outer_xmax, inner_ymax), (outer_xmax, inner_ymin)],
                # Top-left corner
                [(outer_xmin, outer_ymax), (inner_xmin, outer_ymax), (inner_xmin, inner_ymax), (outer_xmin, inner_ymax)],
                # Top-right corner
                [(inner_xmax, outer_ymax), (outer_xmax, outer_ymax), (outer_xmax, inner_ymax), (inner_xmax, inner_ymax)],
                # Bottom-left corner
                [(outer_xmin, outer_ymin), (outer_xmin, inner_ymin), (inner_xmin, inner_ymin), (inner_xmin, outer_ymin)],
                # Bottom-right corner
                [(inner_xmax, outer_ymin), (inner_xmax, inner_ymin), (outer_xmax, inner_ymin), (outer_xmax, outer_ymin)],
            ]
            
            for rect_verts in rects:
                if len(rect_verts) >= 3:
                    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": rect_verts})
                    shader.bind()
                    # Multiply calculated alpha by color alpha (if available)
                    color_alpha = gradient_color[3] if len(gradient_color) > 3 else 1.0
                    final_alpha = alpha * color_alpha
                    shader.uniform_float("color", (gradient_color[0], gradient_color[1], gradient_color[2], final_alpha))
                    batch.draw(shader)
    
    try:
        gpu.state.blend_set('NONE')
    except Exception:
        pass

def _adjust_tolerance_multiplier(current_step, direction):
    """Adjust tolerance multiplier step. Returns new step index and multiplier value.
    
    direction: 1 for increase, -1 for decrease
    Returns: (step_index, multiplier_percent)
    
    Step indexing:
    - step -1 = 100% (base)
    - step -2, -3, -4... = decreasing values (90, 80, 70, ... down to 0.01)
    - step 0, 1, 2... = increasing values (110, 130, 170, ...)
    """
    steps = _get_tolerance_multiplier_steps()
    decreasing_values = _get_decreasing_multiplier_values()
    max_step = len(steps) - 1
    min_step = -(len(decreasing_values) + 1)  # +1 because step -1 is 100%, then -2 onwards are decreasing
    
    if direction > 0:  # Increase
        if current_step < -1:
            # We're in decreasing range (below 100%), move up one step
            # step -2 -> -1 (go to 100%)
            # step -3 -> -2 (go from 80 to 90)
            # etc.
            new_step = current_step + 1
            if new_step == -1:
                # Reached 100%
                return -1, 100.0
            else:
                multiplier = _step_index_to_multiplier(new_step)
                return new_step, multiplier
        elif current_step == -1:
            # At 100%, go to first increasing step (110%)
            new_step = 0
            multiplier = _step_index_to_multiplier(new_step)
            return new_step, multiplier
        else:
            # We're in increasing range, move up one step
            new_step = min(current_step + 1, max_step)
            multiplier = _step_index_to_multiplier(new_step)
            return new_step, multiplier
    else:  # Decrease
        if current_step > 0:
            # We're in increasing range, go down one step at a time (don't snap to 100%)
            new_step = max(0, current_step - 1)
            if new_step == 0:
                # We're at step 0, next step down is 100%
                return -1, 100.0
            else:
                multiplier = _step_index_to_multiplier(new_step)
                return new_step, multiplier
        elif current_step == -1:
            # At 100%, start decreasing (go to 90%)
            new_step = -2
            multiplier = _step_index_to_multiplier(new_step)
            return new_step, multiplier
        else:
            # We're in decreasing range, move down one step
            new_step = max(current_step - 1, min_step)
            multiplier = _step_index_to_multiplier(new_step)
            # Ensure we don't go below 0.01
            multiplier = max(0.01, multiplier)
            return new_step, multiplier

def _matches_light_with_type_check(obj, obj_type, obj_metric, ref_objects, ref_metrics, use_percentage, percent_tolerance, abs_tolerance, tolerance_multiplier=100.0, depsgraph=None, use_evaluated=False):
    """Specialized matching function for lights that checks light type FIRST.
    
    Returns True only if:
    1. Light type matches at least one reference light type
    2. Power (metric) matches within tolerance
    
    This ensures lights of different types NEVER match, even if power is the same.
    """
    if obj_type != 'LIGHT':
        # Not a light, use regular matching
        return _matches_any_reference(obj_type, obj_metric, ref_metrics, use_percentage, percent_tolerance, abs_tolerance, tolerance_multiplier)
    
    # Get object's light type
    obj_light_type = get_light_type(obj)
    if obj_light_type is None:
        return False  # Can't determine light type, don't match
    
    # Check if ANY reference light has the same type
    has_matching_type = False
    matching_ref_metrics = []
    
    # Build a mapping from ref objects to their metrics
    ref_obj_to_metric = {}
    for ref_obj in ref_objects:
        if ref_obj.type == 'LIGHT':
            if depsgraph:
                ref_obj_type, ref_obj_metric = get_object_metric(ref_obj, depsgraph, use_evaluated)
            else:
                ref_obj_type, ref_obj_metric = get_object_metric(ref_obj, None, False)
            if ref_obj_type == 'LIGHT':
                ref_obj_to_metric[ref_obj] = ref_obj_metric
    
    # Find matching reference lights by type, then collect their metrics
    for ref_obj in ref_objects:
        if ref_obj.type != 'LIGHT':
            continue
        ref_obj_light_type = get_light_type(ref_obj)
        if ref_obj_light_type is None:
            continue
        if ref_obj_light_type == obj_light_type:
            has_matching_type = True
            # Get the metric for this matching reference light
            if ref_obj in ref_obj_to_metric:
                ref_metric = ref_obj_to_metric[ref_obj]
                matching_ref_metrics.append(('LIGHT', ref_metric))
    
    # CRITICAL: If no reference light has the same type, return False immediately
    if not has_matching_type or not matching_ref_metrics:
        return False
    
    # Now check power matching using only the matching reference metrics
    return _matches_any_reference(obj_type, obj_metric, matching_ref_metrics, use_percentage, percent_tolerance, abs_tolerance, tolerance_multiplier)


def _matches_text_with_content_check(obj, obj_type, obj_metric, ref_objects, ref_metrics, use_percentage, percent_tolerance, abs_tolerance, tolerance_multiplier=100.0, depsgraph=None, use_evaluated=False):
    """Specialized matching function for text objects that checks text content FIRST.
    
    Returns True only if:
    1. Text content matches at least one reference text content
    2. Metric matches within tolerance
    
    This ensures text objects with different content NEVER match, even if metric is the same.
    """
    if obj_type != 'FONT':
        # Not a text object, use regular matching
        return _matches_any_reference(obj_type, obj_metric, ref_metrics, use_percentage, percent_tolerance, abs_tolerance, tolerance_multiplier)
    
    # Get object's text content
    obj_text_content = get_text_content(obj)
    if obj_text_content is None:
        return False  # Can't determine text content, don't match
    
    # Check if ANY reference text has the same content
    has_matching_content = False
    matching_ref_metrics = []
    
    # Build a mapping from ref objects to their metrics
    ref_obj_to_metric = {}
    for ref_obj in ref_objects:
        if ref_obj.type == 'FONT':
            if depsgraph:
                ref_obj_type, ref_obj_metric = get_object_metric(ref_obj, depsgraph, use_evaluated)
            else:
                ref_obj_type, ref_obj_metric = get_object_metric(ref_obj, None, False)
            if ref_obj_type == 'FONT':
                ref_obj_to_metric[ref_obj] = ref_obj_metric
    
    # Find matching reference texts by content, then collect their metrics
    for ref_obj in ref_objects:
        if ref_obj.type != 'FONT':
            continue
        ref_obj_text_content = get_text_content(ref_obj)
        if ref_obj_text_content is None:
            continue
        if ref_obj_text_content == obj_text_content:
            has_matching_content = True
            if ref_obj in ref_obj_to_metric:
                matching_ref_metrics.append(('FONT', ref_obj_to_metric[ref_obj]))
    
    # If no matching text content found, don't match
    if not has_matching_content:
        return False
    
    # Now check if metric matches any of the matching reference metrics
    return _matches_any_reference(obj_type, obj_metric, matching_ref_metrics, use_percentage, percent_tolerance, abs_tolerance, tolerance_multiplier)


def _matches_any_reference(obj_type, obj_metric, ref_metrics, use_percentage, percent_tolerance, abs_tolerance, tolerance_multiplier=100.0):
    """Return True if (obj_type,obj_metric) matches any of the reference metrics.

    ref_metrics: list of (type, metric) tuples
    tolerance_multiplier: percentage multiplier (100.0 = 100%, 110.0 = 110%, etc.)
    
    When multiplier < 100%: Selects all objects in range from ref down to scaled ref.
        Example: ref=100, multiplier=4% -> selects objects with 4 to 100 vertices.
    When multiplier == 100%: Matches objects similar to reference (original behavior).
    When multiplier > 100%: Selects all objects in range from ref up to scaled ref.
        Example: ref=100, multiplier=150% -> selects objects with 100 to 150 vertices.
    """
    if not ref_metrics:
        return False

    multiplier_factor = tolerance_multiplier / 100.0

    for ref_type, ref_metric in ref_metrics:
        # if not allowed cross-types, the caller will filter by type; here we only compare metrics
        if ref_metric == 0.0 and obj_metric == 0.0:
            return True
        
        if tolerance_multiplier < 100.0:
            # When multiplier < 100%, select all objects in range from ref down to scaled ref
            # Example: ref=100, multiplier=4% (scaled=4) -> select all objects from 4 to 100 vertices
            scaled_ref_metric = ref_metric * multiplier_factor
            
            # Determine the range bounds (ref is always >= scaled when multiplier < 100%)
            min_metric = scaled_ref_metric
            max_metric = ref_metric
            
            # Apply tolerance only at the lower boundary (allow slightly below scaled value)
            if use_percentage and scaled_ref_metric:
                lower_tolerance = scaled_ref_metric * (percent_tolerance / 100.0)
            else:
                lower_tolerance = float(abs_tolerance)
            
            # Check if object metric falls within the range: [scaled - tolerance, ref]
            # This selects everything from ref down to scaled (with small tolerance below scaled)
            if min_metric - lower_tolerance <= obj_metric <= max_metric:
                return True
        else:
            # When multiplier >= 100%
            if tolerance_multiplier == 100.0:
                # At 100%, use original behavior: match objects similar to reference
                if use_percentage and ref_metric:
                    allowed = ref_metric * (percent_tolerance / 100.0)
                else:
                    allowed = float(abs_tolerance)
                if abs(obj_metric - ref_metric) <= allowed:
                    return True
            else:
                # When multiplier > 100%, select all objects in range from ref up to scaled ref
                # Example: ref=100, multiplier=150% (scaled=150) -> select objects with 100 to 150 vertices
                scaled_ref_metric = ref_metric * multiplier_factor
                
                # Determine the range bounds (ref is always <= scaled when multiplier > 100%)
                min_metric = ref_metric
                max_metric = scaled_ref_metric
                
                # Apply tolerance only at the upper boundary (allow slightly above scaled value)
                if use_percentage and scaled_ref_metric:
                    upper_tolerance = scaled_ref_metric * (percent_tolerance / 100.0)
                else:
                    upper_tolerance = float(abs_tolerance)
                
                # Check if object metric falls within the range: [ref, scaled + tolerance]
                # This selects everything from ref up to scaled (with small tolerance above scaled)
                if min_metric <= obj_metric <= max_metric + upper_tolerance:
                    return True
    return False


def _is_curve_edit_mode(context):
    """Return True if the user is currently editing a curve object."""
    # Blender reports edit modes like 'EDIT_CURVE'. Also check active_object type to be safe.
    mode = getattr(context, "mode", "")
    ao = context.active_object
    return (mode == 'EDIT_CURVE') or (mode.startswith('EDIT') and ao is not None and ao.type == 'CURVE')


class VIEW3D_OT_box_select_similar_verts(bpy.types.Operator):
    """Press hotkey to arm; left-click to start box, move, release to finish selection."""
    bl_idname = "view3d.select_box_similar_verts"
    bl_label = "Box Select: Similar Vertex Count"
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}

    tolerance: IntProperty(
        name="Absolute tolerance",
        default=0,
        min=0,
        description="Allowed absolute difference in object metric (if Use Percentage is off). Default 0 = exact match only."
    )
    use_percentage: BoolProperty(
        name="Use Percentage",
        default=False,
        description="Interpret tolerance as percentage of reference object's metric"
    )
    percent_tolerance: FloatProperty(
        name="Percent tolerance",
        default=0.0,
        min=0.0,
        max=100.0,
        description="Allowed percentage difference (if Use Percentage on). Default 0.0 = exact match only."
    )
    use_evaluated: BoolProperty(
        name="Use evaluated data (modifiers)",
        default=False,
        description="Count vertices after modifiers are applied (may be slower for meshes)"
    )
    include_other_types: BoolProperty(
        name="Include other object types",
        default=False,
        description="Allow matching objects of different Blender types (e.g. match Empties to Meshes) using the computed metric"
    )
    match_all_selected: BoolProperty(
        name="Match to all selected",
        default=True,
        description="Match candidates against ANY of the currently selected objects (instead of only the active object)"
    )


    def draw_callback(self, context):
        # Safety check: verify operator instance is still valid
        try:
            # Try to access a simple attribute to check if operator is still alive
            _ = self.bl_idname
        except (ReferenceError, AttributeError):
            # Operator has been removed, don't draw anything
            return
        
        try:
            region = context.region

            # Draw crosshair lines only when armed (before drawing starts)
            current_mouse = _safe_getattr(self, "current_mouse", None)
            drawing = _safe_getattr(self, 'drawing', False)
            
            if current_mouse is not None and not drawing:
                x, y = current_mouse

                # Create crosshair lines extending to viewport edges (solid for this operator)
                crosshair_verts = [
                    (x, 0), (x, region.height),
                    (0, y), (region.width, y)
                ]

                try:
                    gpu.state.blend_set('ALPHA')
                except Exception:
                    pass

                # Get outline color from preferences
                prefs = _get_addon_preferences()
                outline_color = tuple(prefs.selection_outline_color) if prefs else (1.0, 1.0, 1.0, 1.0)
                # Ensure 4 components (RGBA) - handle old 3-component colors
                if len(outline_color) == 3:
                    outline_color = outline_color + (1.0,)

                shader = gpu.shader.from_builtin('UNIFORM_COLOR')
                batch = batch_for_shader(shader, 'LINES', {"pos": crosshair_verts})
                shader.bind()
                color_alpha = outline_color[3] if len(outline_color) > 3 else 1.0
                shader.uniform_float("color", (outline_color[0], outline_color[1], outline_color[2], 0.6 * color_alpha))
                batch.draw(shader)

                try:
                    gpu.state.blend_set('NONE')
                except Exception:
                    pass

            # Draw selection box if we're dragging (handles both LMB selection and MMB-deselect in custom modal)
            start = _safe_getattr(self, "start", None)
            end = _safe_getattr(self, "end", None)
            
            if start is None or end is None:
                return

            x1, y1 = self.start
            x2, y2 = self.end

            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            verts = [
                (float(xmin), float(ymin)),
                (float(xmax), float(ymin)),
                (float(xmax), float(ymax)),
                (float(xmin), float(ymax)),
            ]

            try:
                gpu.state.blend_set('ALPHA')
            except Exception:
                pass

            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            # Filled translucent rect
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
            shader.bind()
            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.02))
            batch.draw(shader)

            # Get outline color from preferences
            prefs = _get_addon_preferences()
            outline_color = tuple(prefs.selection_outline_color) if prefs else (1.0, 1.0, 1.0)
            # Ensure 4 components (RGBA) - handle old 3-component colors
            if len(outline_color) == 3:
                outline_color = outline_color + (1.0,)
            
            # Check if animated zebra mode is active
            if getattr(self, 'animated_zebra_mode', False):
                # Draw animated zebra dashes with glow effect
                try:
                    gpu.state.line_width_set(1.5)
                except Exception:
                    pass
                
                # Get animated zebra specific settings from preferences
                anim_dash_len = prefs.animated_zebra_dash_length if prefs else 5.0
                anim_gap_len = prefs.animated_zebra_gap_length if prefs else 4.0
                # Use default values for glow and gradient (no UI settings needed)
                glow_intensity_base = 0.7
                gradient_dist = 2.0
                travel_cycles = 1.0  # Fixed to 1.0, no UI setting needed
                
                # Calculate animation offset based on time for smooth continuous movement
                current_time = time.time()
                # Get animation speed from preferences
                animation_speed = prefs.animated_zebra_speed if prefs else 40.0
                pattern_length = anim_dash_len + anim_gap_len
                
                # Calculate pulsing glow intensity (sine wave for smooth up/down)
                glow_pulse_speed = 2.0  # cycles per second
                glow_pulse = (math.sin(current_time * glow_pulse_speed * 2 * math.pi) + 1.0) / 2.0  # 0.0 to 1.0
                # Map to desired glow range based on preference
                glow_intensity = 0.3 + (glow_pulse * 0.7 * glow_intensity_base)
                
                # Calculate edge lengths for seamless continuous animation around perimeter
                top_length = xmax - xmin
                right_length = ymax - ymin
                bottom_length = xmax - xmin
                left_length = ymax - ymin
                total_perimeter = top_length + right_length + bottom_length + left_length
                
                # Use a continuous offset for seamless animation
                # Calculate raw continuous offset (NO wrapping) for truly smooth phase calculation
                # Keeping it unwrapped ensures perfect continuity without any jumps
                # Double the speed factor to match expected animation speed
                continuous_offset = current_time * animation_speed * 2
                
                # Calculate normalized offset for edge position determination only
                # This is separate from phase calculation to maintain continuity
                if total_perimeter > 0:
                    normalized_global = continuous_offset % total_perimeter
                else:
                    normalized_global = 0.0
                
                # Build animated dashed segments for all four sides with continuous offset
                # Each side calculates its local offset based on global_offset position in the perimeter
                # This ensures smooth continuous animation around the entire box
                box_dash_segments = []
                
                # Helper function to calculate local offset for an edge
                # This ensures smooth continuous animation with proper wrap-around handling
                # For clockwise motion: top->right, right->down, bottom->left, left->up
                def calculate_edge_offset(edge_start_pos, reverse_direction=False):
                    """Calculate the local pattern offset for an edge given the continuous offset.
                    The speed remains constant - travel_cycles only affects wrap distance, not speed.
                    reverse_direction: if True, reverses the offset direction for edges going backwards."""
                    if pattern_length > 0:
                        # Calculate the continuous offset at this edge's start position
                        # Use subtraction for clockwise motion (offset is negated in drawing function)
                        # For reverse edges (bottom, right), we need to reverse the offset
                        base_offset = edge_start_pos - continuous_offset
                        if reverse_direction:
                            # Reverse the offset direction for edges that go backwards
                            phase_at_edge_start = -base_offset
                        else:
                            phase_at_edge_start = base_offset
                    else:
                        phase_at_edge_start = 0
                    return phase_at_edge_start
                
                # Top edge (left to right) - starts at position 0, moves right
                top_start_pos = 0
                top_local_offset = calculate_edge_offset(top_start_pos, reverse_direction=False)
                top_segs = _build_dashed_segments_animated(xmin, ymax, xmax, ymax, anim_dash_len, anim_gap_len, top_local_offset, travel_cycles)
                for seg in top_segs:
                    x0, y0, x1, y1 = seg
                    box_dash_segments.append((x0, y0))
                    box_dash_segments.append((x1, y1))
                
                # Right edge (top to bottom) - starts after top_length, moves down
                right_start_pos = top_length
                right_local_offset = calculate_edge_offset(right_start_pos, reverse_direction=True)
                right_segs = _build_dashed_segments_animated(xmax, ymax, xmax, ymin, anim_dash_len, anim_gap_len, right_local_offset, travel_cycles)
                for seg in right_segs:
                    x0, y0, x1, y1 = seg
                    box_dash_segments.append((x0, y0))
                    box_dash_segments.append((x1, y1))
                
                # Bottom edge (right to left) - starts after top_length + right_length, moves left
                bottom_start_pos = top_length + right_length
                bottom_local_offset = calculate_edge_offset(bottom_start_pos, reverse_direction=True)
                bottom_segs = _build_dashed_segments_animated(xmax, ymin, xmin, ymin, anim_dash_len, anim_gap_len, bottom_local_offset, travel_cycles)
                for seg in bottom_segs:
                    x0, y0, x1, y1 = seg
                    box_dash_segments.append((x0, y0))
                    box_dash_segments.append((x1, y1))
                
                # Left edge (bottom to top) - starts after top_length + right_length + bottom_length, moves up
                left_start_pos = top_length + right_length + bottom_length
                left_local_offset = calculate_edge_offset(left_start_pos, reverse_direction=False)
                left_segs = _build_dashed_segments_animated(xmin, ymin, xmin, ymax, anim_dash_len, anim_gap_len, left_local_offset, travel_cycles)
                for seg in left_segs:
                    x0, y0, x1, y1 = seg
                    box_dash_segments.append((x0, y0))
                    box_dash_segments.append((x1, y1))
                
                # Draw with pulsing glow effect (multiple passes with different opacities and sizes)
                color_alpha = outline_color[3] if len(outline_color) > 3 else 1.0
                
                # Glow effect: draw multiple passes with decreasing opacity and increasing size
                # Base glow passes - intensity will be modulated by glow_intensity
                glow_passes = [
                    (3.0, 0.12),  # Outer glow, very transparent
                    (2.5, 0.20),  # Mid-outer glow
                    (2.0, 0.30),  # Mid glow
                    (1.5, 0.45),  # Inner glow
                ]
                
                for line_width, base_glow_alpha in glow_passes:
                    try:
                        gpu.state.line_width_set(line_width * glow_intensity)
                    except Exception:
                        pass
                    
                    if box_dash_segments:
                        batch = batch_for_shader(shader, 'LINES', {"pos": box_dash_segments})
                        shader.bind()
                        # Modulate glow alpha with pulsing intensity
                        final_alpha = base_glow_alpha * glow_intensity * color_alpha
                        shader.uniform_float("color", (outline_color[0], outline_color[1], outline_color[2], final_alpha))
                        batch.draw(shader)
                
                # Draw main dashes - all with same bright color (no zebra pattern)
                try:
                    gpu.state.line_width_set(1.5)
                except Exception:
                    pass
                
                # Draw gradient glow on both sides of dashes
                # Create gradient by drawing multiple passes with offset positions and decreasing opacity
                gradient_steps = 3  # Number of gradient layers on each side
                gradient_distance = gradient_dist  # Maximum offset distance in pixels (from preferences)
                
                # Draw gradient glow for all dashes (bright)
                if box_dash_segments:
                    for step in range(gradient_steps, 0, -1):  # Draw from outer to inner
                        offset_dist = (gradient_distance * step / gradient_steps) * glow_intensity
                        gradient_alpha = (0.15 * step / gradient_steps) * glow_intensity
                        
                        # Draw gradient on both sides (positive and negative offset)
                        for side in [-1, 1]:  # -1 for one side, +1 for the other
                            # Calculate perpendicular offsets for each line segment
                            gradient_verts = []
                            for i in range(0, len(box_dash_segments), 2):
                                if i + 1 < len(box_dash_segments):
                                    x0, y0 = box_dash_segments[i]
                                    x1, y1 = box_dash_segments[i + 1]
                                    
                                    # Calculate perpendicular direction (normalized)
                                    dx = x1 - x0
                                    dy = y1 - y0
                                    length = math.sqrt(dx * dx + dy * dy)
                                    if length > 0:
                                        # Perpendicular vector (rotate 90 degrees)
                                        perp_x = -dy / length
                                        perp_y = dx / length
                                        
                                        # Offset both points perpendicularly (both sides)
                                        side_offset = offset_dist * side
                                        gradient_verts.append((x0 + perp_x * side_offset, y0 + perp_y * side_offset))
                                        gradient_verts.append((x1 + perp_x * side_offset, y1 + perp_y * side_offset))
                            
                            if gradient_verts:
                                try:
                                    gpu.state.line_width_set(1.5 + (step * 0.3))
                                except Exception:
                                    pass
                                batch = batch_for_shader(shader, 'LINES', {"pos": gradient_verts})
                                shader.bind()
                                final_alpha = gradient_alpha * color_alpha
                                shader.uniform_float("color", (outline_color[0], outline_color[1], outline_color[2], final_alpha))
                                batch.draw(shader)
                    
                    # Draw main dashes (all bright, no alternating colors)
                    try:
                        gpu.state.line_width_set(1.5)
                    except Exception:
                        pass
                    batch = batch_for_shader(shader, 'LINES', {"pos": box_dash_segments})
                    shader.bind()
                    shader.uniform_float("color", (outline_color[0], outline_color[1], outline_color[2], 0.9 * color_alpha))
                    batch.draw(shader)
                
                # Animation redraw is handled by timer in modal function
            else:
                # Standard solid line drawing
                # Use LINE_STRIP and repeat the first vertex to reliably close the loop (LINE_LOOP can be flaky)
                verts_closed = verts + [verts[0]]
                try:
                    # optional: attempt to set a visible line width (may fail silently on some backends)
                    gpu.state.line_width_set(1.0)
                except Exception:
                    pass
                
                batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": verts_closed})
                shader.bind()
                color_alpha = outline_color[3] if len(outline_color) > 3 else 1.0
                shader.uniform_float("color", (outline_color[0], outline_color[1], outline_color[2], 1.0 * color_alpha))
                batch.draw(shader)

            # Draw gradient visualization when tolerance multiplier is not 100% (if enabled)
            # Don't show gradient or percentage when Ctrl mode is active (select all mode doesn't use tolerance)
            try:
                tolerance_multiplier = getattr(self, 'tolerance_multiplier', 100.0)
            except (ReferenceError, AttributeError):
                tolerance_multiplier = 100.0
            ctrl_pressed = getattr(self, '_ctrl_pressed', False)
            show_gradient = prefs.show_gradient if prefs else True
            
            if tolerance_multiplier != 100.0 and show_gradient and not ctrl_pressed:
                _draw_gradient_box(xmin, ymin, xmax, ymax, tolerance_multiplier)

            # Draw tolerance multiplier text in center of box (if enabled)
            show_percentage = prefs.show_percentage_text if prefs else True
            # Calculate center coordinates (always needed for text positioning)
            center_x = (xmin + xmax) / 2.0
            center_y = (ymin + ymax) / 2.0
            
            if tolerance_multiplier != 100.0 and show_percentage and not ctrl_pressed:
                # Get text settings from preferences
                base_font_size = prefs.text_font_size if prefs else 15
                text_size_scale = prefs.text_size_scale if prefs else 0.35
                text_color = prefs.text_color if prefs else (1.0, 1.0, 1.0, 1.0)
                # Ensure 4 components (RGBA) - handle old 3-component colors
                if len(text_color) == 3:
                    text_color = text_color + (1.0,)
                text_color_r, text_color_g, text_color_b = text_color[0], text_color[1], text_color[2]
                text_placement = prefs.text_placement_box if prefs else 'CENTER'
                
                # Calculate box size for dynamic text scaling
                box_width = xmax - xmin
                box_height = ymax - ymin
                box_size = (box_width + box_height) / 2.0  # Average dimension
                
                # Calculate dynamic font size: blend between fixed size and scaled size
                # Scale factor: convert box size to reasonable font size (e.g., 100px box = ~15px font)
                scale_factor = 0.15  # Adjust this to control scaling sensitivity
                scaled_size = box_size * scale_factor
                text_font_size = base_font_size * (1.0 - text_size_scale) + scaled_size * text_size_scale
                text_font_size = max(10, min(50, text_font_size))  # Clamp between 10 and 50
                
                # Calculate opacity based on box size (fade out when too small)
                # Minimum size where text is fully visible: around 40 pixels (for readable text)
                # Below this, fade from 0 (at 0px) to 1.0 (at 40px)
                min_size_threshold = 40.0
                if box_size <= 0:
                    text_alpha = 0.0
                elif box_size >= min_size_threshold:
                    text_alpha = 1.0
                else:
                    # Smooth fade: 0 to 1.0 as size goes from 0 to min_size_threshold
                    # Use a power curve for faster fade (make it disappear quickly when small)
                    normalized_size = box_size / min_size_threshold
                    text_alpha = pow(normalized_size, 2.5)  # Power curve for faster fade
                
                # Don't draw if too transparent
                if text_alpha < 0.01:
                    pass  # Skip drawing
                else:
                    # Get display mode and format text accordingly
                    # If multiple object types are selected, always show percentage
                    force_percentage = _has_multiple_object_types(self)
                    text_display_mode = prefs.text_display_mode if prefs else 'PERCENTAGE'
                    if text_display_mode == 'VERTEX_COUNT' and not force_percentage:
                        try:
                            threshold_value, obj_type = _get_vertex_count_threshold(self)
                            if obj_type == 'LIGHT':
                                text = _format_power_display(threshold_value) if threshold_value > 0 else "0W"
                            else:
                                text = _format_vertex_count_display(threshold_value) if threshold_value > 0 else "0"
                        except (ReferenceError, AttributeError):
                            text = "0"
                    else:
                        text = _format_multiplier_display(tolerance_multiplier)
                    
                    font_id = 0
                    blf.size(font_id, int(text_font_size))
                    
                    # Get text dimensions first
                    try:
                        text_width, text_height = blf.dimensions(font_id, text)
                    except (AttributeError, TypeError):
                        # Fallback: approximate dimensions based on font size and character count
                        text_width = text_font_size * 0.6 * len(text)
                        text_height = text_font_size * 1.2
                    
                    # Calculate text position based on placement setting
                    if text_placement == 'CENTER':
                        # Center text: blf.position sets bottom-left, so we need to offset by half text dimensions
                        text_x = center_x - text_width / 2.0
                        text_y = center_y - text_height / 2.0
                    else:  # MOUSE
                        # Position at mouse cursor (current center for box is already centered)
                        text_x = center_x
                        text_y = center_y
                    
                    blf.position(font_id, text_x, text_y, 0)
                    # Multiply calculated alpha by color alpha (if available)
                    color_alpha = text_color[3] if len(text_color) > 3 else 1.0
                    final_text_alpha = text_alpha * color_alpha
                    blf.color(font_id, text_color_r, text_color_g, text_color_b, final_text_alpha)
                    blf.draw(font_id, text)

            try:
                gpu.state.blend_set('NONE')
            except Exception:
                pass
        except (ReferenceError, AttributeError):
            # Operator instance has been removed, silently return
            return

    def invoke(self, context, event):
        global _last_custom_modal, _last_toggle_modal

        # Only work in OBJECT mode - otherwise pass through to Blender's default box select
        if context.mode != 'OBJECT':
            override = _find_3dview_override()
            if override is not None:
                try:
                    bpy.ops.view3d.select_box(override, 'INVOKE_DEFAULT')
                except Exception:
                    try:
                        bpy.ops.view3d.select_box('INVOKE_DEFAULT')
                    except Exception:
                        pass
            else:
                try:
                    bpy.ops.view3d.select_box('INVOKE_DEFAULT')
                except Exception:
                    pass
            return {'CANCELLED'}

        # --- NEW: if user is editing a curve, do nothing special — call Blender's builtin and exit ---
        if _is_curve_edit_mode(context):
            override = _find_3dview_override()
            if override is not None:
                try:
                    bpy.ops.view3d.select_box(override, 'INVOKE_DEFAULT')
                except Exception:
                    try:
                        bpy.ops.view3d.select_box('INVOKE_DEFAULT')
                    except Exception:
                        pass
            else:
                try:
                    bpy.ops.view3d.select_box('INVOKE_DEFAULT')
                except Exception:
                    pass
            return {'CANCELLED'}
        # --- end NEW check ---

        if context.area.type != 'VIEW_3D':
            return {'CANCELLED'}

        # Gather reference objects: either all selected (in view layer) or the active object as fallback
        refs = [o for o in context.selected_objects if o.name in context.view_layer.objects]
        if not refs and context.active_object and context.active_object.name in context.view_layer.objects:
            refs = [context.active_object]
        if not refs:
            return {'CANCELLED'}

        # If there is a waiting modal, clear it (we are switching into custom)
        if _last_toggle_modal is not None:
            try:
                _last_toggle_modal._should_cancel = True
                if hasattr(_last_toggle_modal, "_handler") and _last_toggle_modal._handler is not None:
                    try:
                        bpy.types.SpaceView3D.draw_handler_remove(_last_toggle_modal._handler, 'WINDOW')
                    except Exception:
                        pass
                    _last_toggle_modal._handler = None
            except Exception:
                pass
            _last_toggle_modal = None

        # armed but not drawing until first click
        self.start = None
        self.end = None
        self.current_mouse = (event.mouse_region_x, event.mouse_region_y)
        self.drawing = False
        self._deselect_mode = False
        self._handler = None
        self._statusbar_handler = None  # Handler for status bar hints
        self.depsgraph = context.evaluated_depsgraph_get()
        self.animated_zebra_mode = False  # Flag for animated zebra mode
        self._last_shift_press_time = 0.0  # Track last shift press for double-shift detection
        self._animation_timer = None  # Timer handle for continuous redraw
        global _last_tolerance_multiplier_step
        self.tolerance_multiplier_step = _last_tolerance_multiplier_step  # Use remembered value
        self.tolerance_multiplier = _step_index_to_multiplier(self.tolerance_multiplier_step)
        self._prev_box_bounds = None  # Track previous frame's box bounds for deselection
        global _global_ctrl_pressed
        self._ctrl_pressed = _global_ctrl_pressed  # Use global state to persist between invocations
        
        # Throttling variables for mouse movement to prevent lag
        self._last_selection_update_time = 0.0  # Time of last selection update
        self._last_selection_mouse_pos = None  # Last mouse position when selection was updated
        self._selection_throttle_ms = 16.0  # Minimum time between selection updates (ms) - ~60fps
        self._selection_min_move_pixels = 5.0  # Minimum mouse movement (pixels) to force update
        
        # Store initial selection state - objects selected at start should not be deselected
        self._initial_selected = set()
        view_layer_objects = context.view_layer.objects
        for obj in view_layer_objects:
            if obj.select_get():
                self._initial_selected.add(obj.name)

        # compute reference metrics now and store for later matching
        self._ref_metrics = []
        self._ref_objects = []  # Store reference objects directly for light type checking
        self._ref_light_types = {}  # Store light types for reference objects: {ref_obj_name: light_type}
        self._ref_metric_to_obj = {}  # Map (type, metric) to list of reference objects for light type checking
        for o in refs:
            t, m = get_object_metric(o, self.depsgraph, self.use_evaluated)
            self._ref_metrics.append((t, m))
            self._ref_objects.append(o)  # Store reference object directly
            # Store mapping from metric to objects
            metric_key = (t, round(m, 6))
            if metric_key not in self._ref_metric_to_obj:
                self._ref_metric_to_obj[metric_key] = []
            self._ref_metric_to_obj[metric_key].append(o)
            # Store light type if it's a light
            if t == 'LIGHT':
                light_type = get_light_type(o)
                if light_type:
                    self._ref_light_types[o.name] = light_type
            # Store text content if it's a text object
            if t == 'FONT':
                text_content = get_text_content(o)
                if text_content is not None:
                    if not hasattr(self, '_ref_text_contents'):
                        self._ref_text_contents = {}
                    self._ref_text_contents[o.name] = text_content
        
        # Store active object's metric separately for deselection (deselect only based on active object)
        self._active_ref_metrics = []
        self._active_ref_objects = []  # Store active reference object directly for light type and text content checking
        self._active_ref_light_types = {}  # Store light types for active object
        self._active_ref_text_contents = {}  # Store text contents for active object
        self._active_metric_to_obj = {}  # Map (type, metric) to list of active reference objects for light type checking
        if context.active_object and context.active_object.name in context.view_layer.objects:
            t, m = get_object_metric(context.active_object, self.depsgraph, self.use_evaluated)
            self._active_ref_metrics.append((t, m))
            self._active_ref_objects.append(context.active_object)  # Store active object directly
            # Store mapping from metric to objects
            metric_key = (t, round(m, 6))
            if metric_key not in self._active_metric_to_obj:
                self._active_metric_to_obj[metric_key] = []
            self._active_metric_to_obj[metric_key].append(context.active_object)
            # Store light type if it's a light
            if t == 'LIGHT':
                light_type = get_light_type(context.active_object)
                if light_type:
                    self._active_ref_light_types[context.active_object.name] = light_type
            # Store text content if it's a text object
            if t == 'FONT':
                text_content = get_text_content(context.active_object)
                if text_content is not None:
                    self._active_ref_text_contents[context.active_object.name] = text_content

        # Add draw handler immediately to show crosshair (solid)
        self._handler = bpy.types.SpaceView3D.draw_handler_add(self.draw_callback, (context,), 'WINDOW', 'POST_PIXEL')
        
        # Add status bar draw handler to show hints with icons
        # Create a closure that captures the operator instance
        operator_instance = self
        def draw_statusbar_box(self, context):
            """Draw status bar hints with icons for box select"""
            # Check if this operator is still active
            global _last_custom_modal
            if _last_custom_modal != operator_instance:
                return  # Only draw if this is the active operator
            layout = self.layout
            row = layout.row(align=True)
            row.scale_x = 0.0  # Compact spacing
            
            # Shift+Scroll: Adjust tolerance
            row.label(text="", icon='EVENT_SHIFT')
            row.label(text="+")
            row.label(text="", icon='MOUSE_MMB')
            row.label(text=": Adjust tolerance")
            
            # Alt: Reset to 100%
            row.label(text="", icon='EVENT_ALT')
            row.label(text=": Reset to 100%")
            
            # Ctrl: Select all same type
            row.label(text="", icon='EVENT_CTRL')
            row.label(text=": Select all same type")
        
        try:
            # Use prepend to add to left side instead of append (which adds to right side)
            self._statusbar_handler = bpy.types.STATUSBAR_HT_header.prepend(draw_statusbar_box)
        except Exception as e:
            print(f"Error adding status bar handler: {e}")
            pass
        
        context.area.tag_redraw()

        # remember the custom modal instance so pressing B will toggle back
        _last_custom_modal = self

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        global _last_custom_modal, _last_tolerance_multiplier_step, _global_ctrl_pressed

        # If user presses B while this custom modal is active -> cancel it and go to waiting-mode
        if event.type == 'B' and event.value == 'PRESS':
            # cleanup this modal
            if self._handler is not None:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(self._handler, 'WINDOW')
                except Exception:
                    pass
                self._handler = None
            # Remove status bar handler
            if self._statusbar_handler is not None:
                try:
                    bpy.types.STATUSBAR_HT_header.remove(self._statusbar_handler)
                except:
                    pass
                self._statusbar_handler = None
            # Stop animation timer if active
            self.animated_zebra_mode = False  # This will cause the timer to stop itself
            _last_custom_modal = None
            context.area.tag_redraw()
            # invoke waiting-mode (zebra)
            try:
                bpy.ops.view3d.box_select_toggle('INVOKE_DEFAULT')
            except Exception:
                pass
            return {'CANCELLED'}

        # Middle mouse press starts a deselect box (so user can drag to deselect objects) in custom modal
        if event.type == 'MIDDLEMOUSE' and event.value == 'PRESS' and not self.drawing:
            self.start = (event.mouse_region_x, event.mouse_region_y)
            self.end = self.start
            self.current_mouse = self.start
            self.drawing = True
            self._deselect_mode = True
            self._prev_box_bounds = None  # Reset previous box bounds when starting new deselection
            # Track objects that were deselected and their previous selection state (for restoration)
            self._deselected_objects = {}  # {obj_name: was_selected_before}
            # Reset throttling to ensure first update happens immediately
            self._last_selection_update_time = 0.0
            self._last_selection_mouse_pos = None
            
            # Pre-cache object metrics for all visible objects to avoid recalculating during drag
            # This is a major performance optimization when many objects are selected
            self._object_metrics_cache = {}  # {obj_name: (obj_type, obj_metric)}
            region = context.region
            rv3d = context.region_data
            valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
            view_layer_objects = context.view_layer.objects
            
            for obj in view_layer_objects:
                if obj.type not in valid_types:
                    continue
                if obj.hide_viewport:
                    continue
                # Cache the metric once - it won't change during the drag operation
                obj_type, obj_metric = get_object_metric(obj, self.depsgraph, self.use_evaluated)
                self._object_metrics_cache[obj.name] = (obj_type, obj_metric)
            
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # If MMB released while in deselect mode -> perform deselect-in-box
        if event.type == 'MIDDLEMOUSE' and event.value == 'RELEASE' and self.drawing and self._deselect_mode:
            x1, y1 = self.start
            x2, y2 = self.end
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)

            region = context.region
            rv3d = context.region_data

            # Valid object types
            valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
            # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
            view_layer_objects = context.view_layer.objects

            for obj in view_layer_objects:
                if obj.type not in valid_types:
                    continue
                if obj.hide_viewport:
                    continue

                co_world = obj.matrix_world.translation
                co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                if co_2d is None:
                    continue
                x, y = co_2d
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    obj_type, obj_metric = get_object_metric(obj, self.depsgraph, self.use_evaluated)

                    # Use active object's metrics for deselection (not all selected objects)
                    if not self._active_ref_metrics:
                        continue  # No active object to deselect against
                    
                    # For lights, use specialized matching function that checks light type FIRST
                    if obj_type == 'LIGHT' and not self._ctrl_pressed:
                        # Use specialized function that checks light type before power
                        matches = _matches_light_with_type_check(obj, obj_type, obj_metric, self._active_ref_objects, self._active_ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                        if matches:
                            # When user explicitly uses middle mouse, they can deselect anything
                            try:
                                obj.select_set(False)
                            except Exception:
                                pass
                        continue
                    else:
                        # For non-lights or Ctrl mode, use normal filtering
                        # if types must match and this object type isn't among active ref's types, skip
                        if not self.include_other_types:
                            # check whether active ref has same type
                            if not any(r_type == obj_type for r_type, _ in self._active_ref_metrics):
                                continue
                            # compute refs with the same type only for matching
                            refs_to_check = [(r_type, r_metric) for r_type, r_metric in self._active_ref_metrics if r_type == obj_type]
                        else:
                            # Use cached list instead of creating new one
                            if getattr(self, '_active_ref_metrics_list_cache', None) is None:
                                self._active_ref_metrics_list_cache = list(self._active_ref_metrics)
                            refs_to_check = self._active_ref_metrics_list_cache
                        
                        # When user explicitly uses middle mouse, they can deselect anything
                        if _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier):
                            try:
                                obj.select_set(False)
                            except Exception:
                                pass

            # cleanup and finish
            if self._handler is not None:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(self._handler, 'WINDOW')
                except Exception:
                    pass
                self._handler = None

            # Remove status bar handler
            if self._statusbar_handler is not None:
                try:
                    bpy.types.STATUSBAR_HT_header.remove(self._statusbar_handler)
                except:
                    pass
                self._statusbar_handler = None
            
            # Stop animation timer if active
            self.animated_zebra_mode = False  # This will cause the timer to stop itself
            _last_custom_modal = None
            context.area.tag_redraw()
            return {'FINISHED'}

        # cancel on Esc / RightMouse press
        if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            if self._handler is not None:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(self._handler, 'WINDOW')
                except Exception:
                    pass
                self._handler = None
            # Remove status bar handler
            if self._statusbar_handler is not None:
                try:
                    bpy.types.STATUSBAR_HT_header.remove(self._statusbar_handler)
                except:
                    pass
                self._statusbar_handler = None
            # Stop animation timer if active
            self.animated_zebra_mode = False  # This will cause the timer to stop itself
            _last_custom_modal = None
            context.area.tag_redraw()
            return {'CANCELLED'}

        # start on left press (arm drawing)
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and not self.drawing:
            self.start = (event.mouse_region_x, event.mouse_region_y)
            self.end = self.start
            self.current_mouse = self.start
            self.drawing = True
            self._deselect_mode = False
            self._prev_box_bounds = None  # Reset previous box bounds when starting new selection
            # Reset throttling to ensure first update happens immediately
            self._last_selection_update_time = 0.0
            self._last_selection_mouse_pos = None
            
            # Pre-cache object metrics for all visible objects to avoid recalculating during drag
            # This is a major performance optimization when many objects are selected
            self._object_metrics_cache = {}  # {obj_name: (obj_type, obj_metric)}
            region = context.region
            rv3d = context.region_data
            valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
            view_layer_objects = context.view_layer.objects
            
            for obj in view_layer_objects:
                if obj.type not in valid_types:
                    continue
                if obj.hide_viewport:
                    continue
                # Cache the metric once - it won't change during the drag operation
                obj_type, obj_metric = get_object_metric(obj, self.depsgraph, self.use_evaluated)
                self._object_metrics_cache[obj.name] = (obj_type, obj_metric)
            
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Handle Ctrl key press - toggle mode (press to toggle on/off)
        if event.type in {'LEFT_CTRL', 'RIGHT_CTRL'} and event.value == 'PRESS':
            # Toggle Ctrl mode
            _global_ctrl_pressed = not _global_ctrl_pressed
            self._ctrl_pressed = _global_ctrl_pressed
            
            # When Ctrl mode is activated, tolerance adjustment is disabled but the value is remembered
            # (Ctrl mode selects all, so tolerance filtering is not used, but we keep the user's setting)
            
            # Immediately re-evaluate all objects in current selection box (if drawing)
            if self.drawing and hasattr(self, 'start') and hasattr(self, 'end') and self.start is not None and self.end is not None:
                x1, y1 = self.start
                x2, y2 = self.end
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                
                region = context.region
                rv3d = context.region_data
                
                # Performance timing
                t_start = time.perf_counter()
                t_coord_conv_total = 0.0
                t_metric_total = 0.0
                t_matching_total = 0.0
                t_selection_total = 0.0
                objects_processed = 0
                objects_in_box = 0
                matching_calls = 0
                selection_calls = 0
                cache_hits = 0
                cache_misses = 0
                
                # Re-evaluate all objects in current box
                # Valid object types
                valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
                # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
                view_layer_objects = context.view_layer.objects
                
                # Use cached metrics if available (major performance optimization)
                use_cache = hasattr(self, '_object_metrics_cache') and self._object_metrics_cache is not None
                
                for obj in view_layer_objects:
                    objects_processed += 1
                    if obj.type not in valid_types:
                        continue
                    if obj.hide_viewport:
                        continue
                    
                    t_coord_start = time.perf_counter()
                    co_world = obj.matrix_world.translation
                    co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                    t_coord_conv_total += time.perf_counter() - t_coord_start
                    
                    if co_2d is None:
                        continue
                    x, y = co_2d
                    if xmin <= x <= xmax and ymin <= y <= ymax:
                        objects_in_box += 1
                        
                        # Use cached metric if available, otherwise calculate (fallback)
                        t_metric_start = time.perf_counter()
                        if use_cache and obj.name in self._object_metrics_cache:
                            obj_type, obj_metric = self._object_metrics_cache[obj.name]
                            cache_hits += 1
                        else:
                            # Fallback: calculate if cache not available
                            obj_type, obj_metric = get_object_metric(obj, self.depsgraph, self.use_evaluated)
                            cache_misses += 1
                        t_metric_total += time.perf_counter() - t_metric_start
                        
                        if self._ctrl_pressed:
                            # Ctrl mode ON: select all same-type objects
                            ref_types = [r_type for r_type, _ in self._ref_metrics]
                            if obj_type in ref_types:
                                t_sel_start = time.perf_counter()
                                try:
                                    obj.select_set(True)
                                    selection_calls += 1
                                except Exception:
                                    pass
                                t_selection_total += time.perf_counter() - t_sel_start
                        else:
                            # Ctrl mode OFF: re-evaluate with normal criteria
                            # For lights, use specialized matching function that checks light type FIRST
                            if obj_type == 'LIGHT' and not self._ctrl_pressed:
                                # Use specialized function that checks light type before power
                                t_match_start = time.perf_counter()
                                matches = _matches_light_with_type_check(obj, obj_type, obj_metric, self._ref_objects, self._ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                matching_calls += 1
                                t_matching_total += time.perf_counter() - t_match_start
                            elif obj_type == 'FONT' and not self._ctrl_pressed:
                                # Use specialized function that checks text content before metric
                                t_match_start = time.perf_counter()
                                matches = _matches_text_with_content_check(obj, obj_type, obj_metric, self._ref_objects, self._ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                matching_calls += 1
                                t_matching_total += time.perf_counter() - t_match_start
                            else:
                                # For non-lights, non-text, or Ctrl mode, use normal filtering
                                if not self.include_other_types:
                                    # Use pre-computed filtered cache instead of list comprehension
                                    if not hasattr(self, '_ref_metrics_by_type') or self._ref_metrics_by_type is None:
                                        # Fallback: build cache on first use
                                        self._ref_metrics_by_type = {}
                                        for r_type, r_metric in self._ref_metrics:
                                            if r_type not in self._ref_metrics_by_type:
                                                self._ref_metrics_by_type[r_type] = []
                                            self._ref_metrics_by_type[r_type].append((r_type, r_metric))
                                    refs_to_check = self._ref_metrics_by_type.get(obj_type, [])
                                    if not refs_to_check:
                                        t_sel_start = time.perf_counter()
                                        try:
                                            obj.select_set(False)
                                            selection_calls += 1
                                        except Exception:
                                            pass
                                        t_selection_total += time.perf_counter() - t_sel_start
                                        continue
                                    t_match_start = time.perf_counter()
                                    matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                    matching_calls += 1
                                    t_matching_total += time.perf_counter() - t_match_start
                                else:
                                    # Use cached list instead of creating new one
                                    if not hasattr(self, '_ref_metrics_list_cache') or self._ref_metrics_list_cache is None:
                                        self._ref_metrics_list_cache = list(self._ref_metrics)
                                    refs_to_check = self._ref_metrics_list_cache
                                    t_match_start = time.perf_counter()
                                    matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                    matching_calls += 1
                                    t_matching_total += time.perf_counter() - t_match_start
                            
                            t_sel_start = time.perf_counter()
                            try:
                                obj.select_set(matches)
                                selection_calls += 1
                            except Exception:
                                pass
                            t_selection_total += time.perf_counter() - t_sel_start
                
                t_total = time.perf_counter() - t_start
                t_unaccounted = t_total - t_coord_conv_total - t_metric_total - t_matching_total - t_selection_total
                
                # Report performance metrics
                mode_str = "ON" if self._ctrl_pressed else "OFF"
                print(f"[B Selection - CTRL Toggle ({mode_str})] Total: {t_total*1000:.2f}ms | "
                      f"Objects: {objects_processed} processed, {objects_in_box} in box | "
                      f"Coord conv: {t_coord_conv_total*1000:.2f}ms | "
                      f"Get metrics: {t_metric_total*1000:.2f}ms ({cache_hits} cache hits, {cache_misses} misses) | "
                      f"Matching: {t_matching_total*1000:.2f}ms ({matching_calls} calls) | "
                      f"Selection: {t_selection_total*1000:.2f}ms ({selection_calls} calls) | "
                      f"Unaccounted: {t_unaccounted*1000:.2f}ms")
                
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Handle Alt key to reset tolerance multiplier back to 100%
        if event.type == 'LEFT_ALT' or event.type == 'RIGHT_ALT':
            if event.value == 'PRESS':
                self.tolerance_multiplier_step = -1  # Reset to 100%
                self.tolerance_multiplier = 100.0
                _last_tolerance_multiplier_step = -1  # Remember the reset
                
                # Live reselection: if we have a selection box, reselect objects with new tolerance (100%)
                if hasattr(self, 'start') and hasattr(self, 'end') and self.start is not None and self.end is not None:
                    x1, y1 = self.start
                    x2, y2 = self.end
                    xmin, xmax = min(x1, x2), max(x1, x2)
                    ymin, ymax = min(y1, y2), max(y1, y2)
                    
                    region = context.region
                    rv3d = context.region_data
                    
                    # Performance timing
                    t_start = time.perf_counter()
                    t_coord_conv_total = 0.0
                    t_metric_total = 0.0
                    t_matching_total = 0.0
                    t_selection_total = 0.0
                    objects_processed = 0
                    objects_in_box = 0
                    matching_calls = 0
                    selection_calls = 0
                    cache_hits = 0
                    cache_misses = 0
                    
                    # Reselect objects in box with 100% tolerance
                    # Valid object types
                    valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
                    # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
                    view_layer_objects = context.view_layer.objects
                    
                    # Use cached metrics if available (major performance optimization)
                    use_cache = hasattr(self, '_object_metrics_cache') and self._object_metrics_cache is not None
                    
                    for obj in view_layer_objects:
                        objects_processed += 1
                        if obj.type not in valid_types:
                            continue
                        if obj.hide_viewport:
                            continue
                        
                        t_coord_start = time.perf_counter()
                        co_world = obj.matrix_world.translation
                        co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                        t_coord_conv_total += time.perf_counter() - t_coord_start
                        
                        if co_2d is None:
                            continue
                        x, y = co_2d
                        if xmin <= x <= xmax and ymin <= y <= ymax:
                            objects_in_box += 1
                            
                            # Use cached metric if available, otherwise calculate (fallback)
                            t_metric_start = time.perf_counter()
                            if use_cache and obj.name in self._object_metrics_cache:
                                obj_type, obj_metric = self._object_metrics_cache[obj.name]
                                cache_hits += 1
                            else:
                                # Fallback: calculate if cache not available
                                obj_type, obj_metric = get_object_metric(obj, self.depsgraph, self.use_evaluated)
                                cache_misses += 1
                            t_metric_total += time.perf_counter() - t_metric_start
                            
                            # Check if Ctrl is held - select all objects of same type regardless of parameters
                            if self._ctrl_pressed:
                                # Check if this object type matches any of the reference object types
                                # Use cached ref types set instead of list comprehension
                                if not hasattr(self, '_ref_types_set') or self._ref_types_set is None:
                                    self._ref_types_set = {r_type for r_type, _ in self._ref_metrics}
                                if obj_type in self._ref_types_set:
                                    t_sel_start = time.perf_counter()
                                    try:
                                        obj.select_set(True)
                                        selection_calls += 1
                                    except Exception:
                                        pass
                                    t_selection_total += time.perf_counter() - t_sel_start
                                continue
                            
                            # For lights, use specialized matching function that checks light type FIRST
                            if obj_type == 'LIGHT' and not self._ctrl_pressed:
                                # Use specialized function that checks light type before power
                                t_match_start = time.perf_counter()
                                matches = _matches_light_with_type_check(obj, obj_type, obj_metric, self._ref_objects, self._ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                matching_calls += 1
                                t_matching_total += time.perf_counter() - t_match_start
                            elif obj_type == 'FONT' and not self._ctrl_pressed:
                                # Use specialized function that checks text content before metric
                                t_match_start = time.perf_counter()
                                matches = _matches_text_with_content_check(obj, obj_type, obj_metric, self._ref_objects, self._ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                matching_calls += 1
                                t_matching_total += time.perf_counter() - t_match_start
                            else:
                                # For non-lights, non-text, or Ctrl mode, use normal filtering
                                if not self.include_other_types:
                                    # Use pre-computed filtered cache instead of list comprehension
                                    if not hasattr(self, '_ref_metrics_by_type') or self._ref_metrics_by_type is None:
                                        # Fallback: build cache on first use
                                        self._ref_metrics_by_type = {}
                                        for r_type, r_metric in self._ref_metrics:
                                            if r_type not in self._ref_metrics_by_type:
                                                self._ref_metrics_by_type[r_type] = []
                                            self._ref_metrics_by_type[r_type].append((r_type, r_metric))
                                    refs_to_check = self._ref_metrics_by_type.get(obj_type, [])
                                    if not refs_to_check:
                                        continue
                                    t_match_start = time.perf_counter()
                                    matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                    matching_calls += 1
                                    t_matching_total += time.perf_counter() - t_match_start
                                else:
                                    # Use cached list instead of creating new one
                                    if not hasattr(self, '_ref_metrics_list_cache') or self._ref_metrics_list_cache is None:
                                        self._ref_metrics_list_cache = list(self._ref_metrics)
                                    refs_to_check = self._ref_metrics_list_cache
                                    t_match_start = time.perf_counter()
                                    matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                    matching_calls += 1
                                    t_matching_total += time.perf_counter() - t_match_start
                            
                            t_sel_start = time.perf_counter()
                            try:
                                obj.select_set(matches)
                                selection_calls += 1
                            except Exception:
                                pass
                            t_selection_total += time.perf_counter() - t_sel_start
                    
                    t_total = time.perf_counter() - t_start
                    t_unaccounted = t_total - t_coord_conv_total - t_metric_total - t_matching_total - t_selection_total
                    
                    # Report performance metrics
                    print(f"[B Selection - ALT Reset] Total: {t_total*1000:.2f}ms | "
                          f"Objects: {objects_processed} processed, {objects_in_box} in box | "
                          f"Coord conv: {t_coord_conv_total*1000:.2f}ms | "
                          f"Get metrics: {t_metric_total*1000:.2f}ms ({cache_hits} cache hits, {cache_misses} misses) | "
                          f"Matching: {t_matching_total*1000:.2f}ms ({matching_calls} calls) | "
                          f"Selection: {t_selection_total*1000:.2f}ms ({selection_calls} calls) | "
                          f"Unaccounted: {t_unaccounted*1000:.2f}ms")
                
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        # Handle Shift + scroll wheel to adjust tolerance multiplier
        if event.shift and (event.type == 'WHEELUPMOUSE' or event.type == 'WHEELDOWNMOUSE') and event.value == 'PRESS':
            # If Ctrl mode is active, turn it off and allow tolerance adjustment
            # (tolerance adjustment is incompatible with "select all" mode)
            if self._ctrl_pressed:
                _global_ctrl_pressed = False
                self._ctrl_pressed = False
            
            direction = 1 if event.type == 'WHEELUPMOUSE' else -1
            old_multiplier = getattr(self, 'tolerance_multiplier', 100.0)
            self.tolerance_multiplier_step, self.tolerance_multiplier = _adjust_tolerance_multiplier(self.tolerance_multiplier_step, direction)
            # Remember the new tolerance step for next time
            _last_tolerance_multiplier_step = self.tolerance_multiplier_step
            
            # Live reselection/redeselection: if we have a selection box, reapply selection/deselection with new tolerance
            if hasattr(self, 'start') and hasattr(self, 'end') and self.start is not None and self.end is not None:
                x1, y1 = self.start
                x2, y2 = self.end
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                
                region = context.region
                rv3d = context.region_data
                
                # If in deselect mode, reapply deselection with new tolerance
                if self._deselect_mode:
                    # First, restore all previously deselected objects
                    for obj_name, was_selected in list(self._deselected_objects.items()):
                        view_layer_objects = context.view_layer.objects
                        obj = view_layer_objects.get(obj_name)
                        if obj is None:
                            continue
                        try:
                            obj.select_set(was_selected)
                        except Exception:
                            pass
                    # Clear the deselected objects tracking
                    self._deselected_objects = {}
                    
                    # Reapply deselection with new tolerance
                    # Valid object types
                    valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
                    # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
                    view_layer_objects = context.view_layer.objects
                    
                    for obj in view_layer_objects:
                        if obj.type not in valid_types:
                            continue
                        if obj.hide_viewport:
                            continue
                        co_world = obj.matrix_world.translation
                        co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                        if co_2d is None:
                            continue
                        x, y = co_2d
                        if xmin <= x <= xmax and ymin <= y <= ymax:
                            obj_type, obj_metric = get_object_metric(obj, self.depsgraph, self.use_evaluated)
                            
                            # Use active object's metrics for deselection
                            if not self._active_ref_metrics:
                                continue  # No active object to deselect against
                            
                            # For lights, use specialized matching function that checks light type FIRST
                            if obj_type == 'LIGHT' and not self._ctrl_pressed:
                                # Use specialized function that checks light type before power
                                matches = _matches_light_with_type_check(obj, obj_type, obj_metric, self._active_ref_objects, self._active_ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                # If object matches smart selection criteria (based on active object), deselect it
                                if matches:
                                    # Store previous selection state before deselecting
                                    if obj.name not in self._deselected_objects:
                                        self._deselected_objects[obj.name] = obj.select_get()
                                    try:
                                        obj.select_set(False)
                                    except Exception:
                                        pass
                            else:
                                # For non-lights or Ctrl mode, use normal filtering
                                if not self.include_other_types:
                                    # Use pre-computed filtered cache instead of list comprehension
                                    if getattr(self, '_active_ref_metrics_by_type', None) is None:
                                        # Fallback: build cache on first use
                                        self._active_ref_metrics_by_type = {}
                                        for r_type, r_metric in self._active_ref_metrics:
                                            if r_type not in self._active_ref_metrics_by_type:
                                                self._active_ref_metrics_by_type[r_type] = []
                                            self._active_ref_metrics_by_type[r_type].append((r_type, r_metric))
                                    refs_to_check = self._active_ref_metrics_by_type.get(obj_type, [])
                                    if not refs_to_check:
                                        continue
                                    matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                else:
                                    # Use cached list instead of creating new one
                                    if getattr(self, '_active_ref_metrics_list_cache', None) is None:
                                        self._active_ref_metrics_list_cache = list(self._active_ref_metrics)
                                    refs_to_check = self._active_ref_metrics_list_cache
                                    matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                
                                # If object matches smart selection criteria (based on active object), deselect it
                                if matches:
                                    # Store previous selection state before deselecting
                                    if obj.name not in self._deselected_objects:
                                        self._deselected_objects[obj.name] = obj.select_get()
                                    try:
                                        obj.select_set(False)
                                    except Exception:
                                        pass
                else:
                    # Normal selection mode: reselect objects in box with new tolerance
                    # Valid object types
                    valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
                    # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
                    view_layer_objects = context.view_layer.objects
                    
                    for obj in view_layer_objects:
                        if obj.type not in valid_types:
                            continue
                        if obj.hide_viewport:
                            continue
                        co_world = obj.matrix_world.translation
                        co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                        if co_2d is None:
                            continue
                        x, y = co_2d
                        if xmin <= x <= xmax and ymin <= y <= ymax:
                            obj_type, obj_metric = get_object_metric(obj, self.depsgraph, self.use_evaluated)
                            
                            # For lights, use specialized matching function that checks light type FIRST
                            if obj_type == 'LIGHT' and not self._ctrl_pressed:
                                # Use specialized function that checks light type before power
                                matches = _matches_light_with_type_check(obj, obj_type, obj_metric, self._ref_objects, self._ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                            elif obj_type == 'FONT' and not self._ctrl_pressed:
                                # Use specialized function that checks text content before metric
                                matches = _matches_text_with_content_check(obj, obj_type, obj_metric, self._ref_objects, self._ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                            else:
                                # For non-lights, non-text, or Ctrl mode, use normal filtering
                                if not self.include_other_types:
                                    # Use pre-computed filtered cache instead of list comprehension
                                    if not hasattr(self, '_ref_metrics_by_type') or self._ref_metrics_by_type is None:
                                        # Fallback: build cache on first use
                                        self._ref_metrics_by_type = {}
                                        for r_type, r_metric in self._ref_metrics:
                                            if r_type not in self._ref_metrics_by_type:
                                                self._ref_metrics_by_type[r_type] = []
                                            self._ref_metrics_by_type[r_type].append((r_type, r_metric))
                                    refs_to_check = self._ref_metrics_by_type.get(obj_type, [])
                                    if not refs_to_check:
                                        continue
                                    matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                else:
                                    # Use cached list instead of creating new one
                                    if not hasattr(self, '_ref_metrics_list_cache') or self._ref_metrics_list_cache is None:
                                        self._ref_metrics_list_cache = list(self._ref_metrics)
                                    refs_to_check = self._ref_metrics_list_cache
                                    matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                            
                            try:
                                obj.select_set(matches)
                            except Exception:
                                pass
            
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Handle Shift-Shift to toggle animated zebra mode
        if event.type in {'LEFT_SHIFT', 'RIGHT_SHIFT'} and event.value == 'PRESS':
            current_time = time.time()
            # Check if this is a double-shift (within 0.5 seconds)
            if current_time - self._last_shift_press_time < 0.5:
                # Toggle animated zebra mode
                self.animated_zebra_mode = not self.animated_zebra_mode
                
                # Manage animation timer
                if self.animated_zebra_mode:
                    # Start animation timer - capture context in closure
                    operator_self = self
                    context_ref = context
                    def animation_timer():
                        try:
                            if hasattr(operator_self, 'animated_zebra_mode') and operator_self.animated_zebra_mode:
                                # Force redraw of all 3D viewports
                                for area in context_ref.screen.areas:
                                    if area.type == 'VIEW_3D':
                                        area.tag_redraw()
                                return 0.016  # ~60fps for smooth animation
                            return None  # Stop timer
                        except:
                            return None  # Stop timer on error
                    # Only register if not already registered
                    if self._animation_timer is None:
                        self._animation_timer = bpy.app.timers.register(animation_timer, first_interval=0.016)
                else:
                    # Stop animation timer (it will stop itself when animated_zebra_mode is False)
                    self._animation_timer = None
                
                context.area.tag_redraw()
            self._last_shift_press_time = current_time
            return {'RUNNING_MODAL'}

        # Force continuous redraw when animated zebra mode is active
        # (Timer handles continuous redraw, but we also redraw on events for responsiveness)
        if getattr(self, 'animated_zebra_mode', False):
            context.area.tag_redraw()

        # update while moving - perform live selection/deselection as user drags
        if event.type == 'MOUSEMOVE':
            self.current_mouse = (event.mouse_region_x, event.mouse_region_y)
            if self.drawing:
                self.end = self.current_mouse
                
                x1, y1 = self.start
                x2, y2 = self.end
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                
                # Check if continuous selection should be disabled (check before processing)
                should_disable, reason = _should_disable_continuous_selection(context)
                if should_disable:
                    # Store flag to disable continuous updates
                    self._continuous_disabled = True
                    if not hasattr(self, '_continuous_disabled_reason') or self._continuous_disabled_reason != reason:
                        print(f"[B Selection] Continuous selection disabled: {reason}")
                        self._continuous_disabled_reason = reason
                    # Skip selection updates, just update visual
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                else:
                    self._continuous_disabled = False
                
                # Throttling: Only process selection if enough time has passed or mouse moved significantly
                current_time = time.perf_counter() * 1000.0  # Convert to milliseconds
                should_process_selection = False
                
                if self._last_selection_update_time == 0.0:
                    # First update - always process
                    should_process_selection = True
                else:
                    time_since_last = current_time - self._last_selection_update_time
                    
                    # Check if enough time has passed
                    if time_since_last >= self._selection_throttle_ms:
                        should_process_selection = True
                    else:
                        # Check if mouse moved significantly (even if time hasn't passed)
                        if self._last_selection_mouse_pos is not None:
                            last_x, last_y = self._last_selection_mouse_pos
                            mouse_dx = abs(x2 - last_x)
                            mouse_dy = abs(y2 - last_y)
                            mouse_distance = math.sqrt(mouse_dx * mouse_dx + mouse_dy * mouse_dy)
                            if mouse_distance >= self._selection_min_move_pixels:
                                should_process_selection = True
                
                # Always update visual, but only process selection if throttling allows
                context.area.tag_redraw()
                
                if not should_process_selection:
                    return {'RUNNING_MODAL'}
                
                # Update throttling state
                self._last_selection_update_time = current_time
                self._last_selection_mouse_pos = (x2, y2)
                
                region = context.region
                rv3d = context.region_data
                
                # Re-evaluate objects in current box based on current Ctrl toggle state
                if not self._deselect_mode:
                    # Performance timing
                    t_start = time.perf_counter()
                    t_coord_conv_total = 0.0
                    t_metric_total = 0.0
                    t_matching_total = 0.0
                    t_selection_total = 0.0
                    objects_processed = 0
                    objects_in_box = 0
                    matching_calls = 0
                    selection_calls = 0
                    cache_hits = 0
                    cache_misses = 0
                    
                    # Valid object types
                    valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
                    # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
                    view_layer_objects = context.view_layer.objects
                    
                    # Use cached metrics if available (major performance optimization)
                    use_cache = hasattr(self, '_object_metrics_cache') and self._object_metrics_cache is not None
                    
                    for obj in view_layer_objects:
                        objects_processed += 1
                        if obj.type not in valid_types:
                            continue
                        if obj.hide_viewport:
                            continue
                        
                        t_coord_start = time.perf_counter()
                        co_world = obj.matrix_world.translation
                        co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                        t_coord_conv_total += time.perf_counter() - t_coord_start
                        
                        if co_2d is None:
                            continue
                        x, y = co_2d
                        if xmin <= x <= xmax and ymin <= y <= ymax:
                            objects_in_box += 1
                            
                            # Use cached metric if available, otherwise calculate (fallback)
                            t_metric_start = time.perf_counter()
                            if use_cache and obj.name in self._object_metrics_cache:
                                obj_type, obj_metric = self._object_metrics_cache[obj.name]
                                cache_hits += 1
                            else:
                                # Fallback: calculate if cache not available
                                obj_type, obj_metric = get_object_metric(obj, self.depsgraph, self.use_evaluated)
                                cache_misses += 1
                            t_metric_total += time.perf_counter() - t_metric_start
                            
                            if self._ctrl_pressed:
                                # Ctrl pressed: select all same-type objects
                                # Use cached ref types set instead of list comprehension
                                if not hasattr(self, '_ref_types_set') or self._ref_types_set is None:
                                    self._ref_types_set = {r_type for r_type, _ in self._ref_metrics}
                                if obj_type in self._ref_types_set:
                                    t_sel_start = time.perf_counter()
                                    try:
                                        obj.select_set(True)
                                        selection_calls += 1
                                    except Exception:
                                        pass
                                    t_selection_total += time.perf_counter() - t_sel_start
                            else:
                                # Ctrl released: re-evaluate with normal criteria
                                if not self.include_other_types:
                                    # Use pre-computed filtered cache instead of list comprehension
                                    if not hasattr(self, '_ref_metrics_by_type') or self._ref_metrics_by_type is None:
                                        # Fallback: build cache on first use
                                        self._ref_metrics_by_type = {}
                                        for r_type, r_metric in self._ref_metrics:
                                            if r_type not in self._ref_metrics_by_type:
                                                self._ref_metrics_by_type[r_type] = []
                                            self._ref_metrics_by_type[r_type].append((r_type, r_metric))
                                    refs_to_check = self._ref_metrics_by_type.get(obj_type, [])
                                    if not refs_to_check:
                                        t_sel_start = time.perf_counter()
                                        try:
                                            obj.select_set(False)
                                            selection_calls += 1
                                        except Exception:
                                            pass
                                        t_selection_total += time.perf_counter() - t_sel_start
                                        continue
                                else:
                                    # Use cached list instead of creating new one
                                    if not hasattr(self, '_ref_metrics_list_cache') or self._ref_metrics_list_cache is None:
                                        self._ref_metrics_list_cache = list(self._ref_metrics)
                                    refs_to_check = self._ref_metrics_list_cache
                                
                                t_match_start = time.perf_counter()
                                matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                matching_calls += 1
                                t_matching_total += time.perf_counter() - t_match_start
                                
                                t_sel_start = time.perf_counter()
                                try:
                                    obj.select_set(matches)
                                    selection_calls += 1
                                except Exception:
                                    pass
                                t_selection_total += time.perf_counter() - t_sel_start
                    
                    t_total = time.perf_counter() - t_start
                    t_unaccounted = t_total - t_coord_conv_total - t_metric_total - t_matching_total - t_selection_total
                    
                    # Report performance metrics
                    print(f"[B Selection - SELECT] Total: {t_total*1000:.2f}ms | "
                          f"Objects: {objects_processed} processed, {objects_in_box} in box | "
                          f"Coord conv: {t_coord_conv_total*1000:.2f}ms | "
                          f"Get metrics: {t_metric_total*1000:.2f}ms ({cache_hits} cache hits, {cache_misses} misses) | "
                          f"Matching: {t_matching_total*1000:.2f}ms ({matching_calls} calls) | "
                          f"Selection: {t_selection_total*1000:.2f}ms ({selection_calls} calls) | "
                          f"Unaccounted: {t_unaccounted*1000:.2f}ms")
                
                # Perform live deselection while dragging with middle mouse
                if self._deselect_mode:
                    # First, restore objects that were deselected but are now outside the box
                    if self._prev_box_bounds is not None:
                        prev_xmin, prev_xmax, prev_ymin, prev_ymax = self._prev_box_bounds
                        view_layer_objects = context.view_layer.objects
                        for obj_name, was_selected in list(self._deselected_objects.items()):
                            obj = view_layer_objects.get(obj_name)
                            if obj is None:
                                continue
                            co_world = obj.matrix_world.translation
                            co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                            if co_2d is None:
                                continue
                            x, y = co_2d
                            # Check if object was in previous box
                            was_in_prev = prev_xmin <= x <= prev_xmax and prev_ymin <= y <= prev_ymax
                            # Check if object is in current box
                            is_in_current = xmin <= x <= xmax and ymin <= y <= ymax
                            # If was in previous but not in current, restore its selection state
                            if was_in_prev and not is_in_current:
                                try:
                                    obj.select_set(was_selected)
                                    # Remove from tracking since it's restored
                                    del self._deselected_objects[obj_name]
                                except Exception:
                                    pass
                    
                    # Track objects currently in box
                    current_box_objects = set()
                    
                    # Performance timing for deselection
                    t_start = time.perf_counter()
                    t_coord_conv_total = 0.0
                    t_metric_total = 0.0
                    t_matching_total = 0.0
                    t_selection_total = 0.0
                    objects_processed = 0
                    objects_in_box = 0
                    matching_calls = 0
                    selection_calls = 0
                    cache_hits = 0
                    cache_misses = 0
                    
                    # Perform smart deselection on objects currently in box
                    # Use only active object as reference for deselection (not all selected objects)
                    # Valid object types
                    valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
                    # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
                    view_layer_objects = context.view_layer.objects
                    
                    # Use cached metrics if available (major performance optimization)
                    use_cache = hasattr(self, '_object_metrics_cache') and self._object_metrics_cache is not None
                    
                    for obj in view_layer_objects:
                        objects_processed += 1
                        if obj.type not in valid_types:
                            continue
                        if obj.hide_viewport:
                            continue
                        
                        t_coord_start = time.perf_counter()
                        co_world = obj.matrix_world.translation
                        co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                        t_coord_conv_total += time.perf_counter() - t_coord_start
                        
                        if co_2d is None:
                            continue
                        x, y = co_2d
                        if xmin <= x <= xmax and ymin <= y <= ymax:
                            current_box_objects.add(obj.name)
                            objects_in_box += 1
                            
                            # Use cached metric if available, otherwise calculate (fallback)
                            t_metric_start = time.perf_counter()
                            if use_cache and obj.name in self._object_metrics_cache:
                                obj_type, obj_metric = self._object_metrics_cache[obj.name]
                                cache_hits += 1
                            else:
                                # Fallback: calculate if cache not available
                                obj_type, obj_metric = get_object_metric(obj, self.depsgraph, self.use_evaluated)
                                cache_misses += 1
                            t_metric_total += time.perf_counter() - t_metric_start
                            
                            # Use active object's metrics for deselection
                            if not self._active_ref_metrics:
                                continue  # No active object to deselect against
                            
                            # For lights, use specialized matching function that checks light type FIRST
                            if obj_type == 'LIGHT' and not self._ctrl_pressed:
                                # Use specialized function that checks light type before power
                                t_match_start = time.perf_counter()
                                matches = _matches_light_with_type_check(obj, obj_type, obj_metric, self._active_ref_objects, self._active_ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                matching_calls += 1
                                t_matching_total += time.perf_counter() - t_match_start
                                
                                # If object matches smart selection criteria (based on active object), deselect it
                                # When user explicitly uses middle mouse, they can deselect anything
                                if matches:
                                    # Store previous selection state before deselecting
                                    if obj.name not in self._deselected_objects:
                                        self._deselected_objects[obj.name] = obj.select_get()
                                    t_sel_start = time.perf_counter()
                                    try:
                                        obj.select_set(False)
                                        selection_calls += 1
                                    except Exception:
                                        pass
                                    t_selection_total += time.perf_counter() - t_sel_start
                            else:
                                # For non-lights or Ctrl mode, use normal filtering
                                if not self.include_other_types:
                                    # Use pre-computed filtered cache instead of list comprehension
                                    if getattr(self, '_active_ref_metrics_by_type', None) is None:
                                        # Fallback: build cache on first use
                                        self._active_ref_metrics_by_type = {}
                                        for r_type, r_metric in self._active_ref_metrics:
                                            if r_type not in self._active_ref_metrics_by_type:
                                                self._active_ref_metrics_by_type[r_type] = []
                                            self._active_ref_metrics_by_type[r_type].append((r_type, r_metric))
                                    refs_to_check = self._active_ref_metrics_by_type.get(obj_type, [])
                                    if not refs_to_check:
                                        continue
                                    t_match_start = time.perf_counter()
                                    matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                    matching_calls += 1
                                    t_matching_total += time.perf_counter() - t_match_start
                                else:
                                    # Use cached list instead of creating new one
                                    if getattr(self, '_active_ref_metrics_list_cache', None) is None:
                                        self._active_ref_metrics_list_cache = list(self._active_ref_metrics)
                                    refs_to_check = self._active_ref_metrics_list_cache
                                    t_match_start = time.perf_counter()
                                    matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                    matching_calls += 1
                                    t_matching_total += time.perf_counter() - t_match_start
                                
                                # If object matches smart selection criteria (based on active object), deselect it
                                # When user explicitly uses middle mouse, they can deselect anything
                                if matches:
                                    # Store previous selection state before deselecting
                                    if obj.name not in self._deselected_objects:
                                        self._deselected_objects[obj.name] = obj.select_get()
                                    t_sel_start = time.perf_counter()
                                    try:
                                        obj.select_set(False)
                                        selection_calls += 1
                                    except Exception:
                                        pass
                                    t_selection_total += time.perf_counter() - t_sel_start
                    
                    t_total = time.perf_counter() - t_start
                    t_unaccounted = t_total - t_coord_conv_total - t_metric_total - t_matching_total - t_selection_total
                    
                    # Report performance metrics
                    print(f"[B Selection - DESELECT] Total: {t_total*1000:.2f}ms | "
                          f"Objects: {objects_processed} processed, {objects_in_box} in box | "
                          f"Coord conv: {t_coord_conv_total*1000:.2f}ms | "
                          f"Get metrics: {t_metric_total*1000:.2f}ms ({cache_hits} cache hits, {cache_misses} misses) | "
                          f"Matching: {t_matching_total*1000:.2f}ms ({matching_calls} calls) | "
                          f"Selection: {t_selection_total*1000:.2f}ms ({selection_calls} calls) | "
                          f"Unaccounted: {t_unaccounted*1000:.2f}ms")
                    
                    # Store current box bounds for next frame
                    self._prev_box_bounds = (xmin, xmax, ymin, ymax)
                
                # Perform live selection while dragging (left mouse)
                else:
                    # First, deselect objects that were in previous box but are now outside
                    if self._prev_box_bounds is not None:
                        prev_xmin, prev_xmax, prev_ymin, prev_ymax = self._prev_box_bounds
                        # Valid object types
                        valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
                        # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
                        view_layer_objects = context.view_layer.objects
                        
                        for obj in view_layer_objects:
                            if obj.type not in valid_types:
                                continue
                            if obj.hide_viewport:
                                continue
                            co_world = obj.matrix_world.translation
                            co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                            if co_2d is None:
                                continue
                            x, y = co_2d
                            # Check if object was in previous box
                            was_in_prev = prev_xmin <= x <= prev_xmax and prev_ymin <= y <= prev_ymax
                            # Check if object is in current box
                            is_in_current = xmin <= x <= xmax and ymin <= y <= ymax
                            # If was in previous but not in current, deselect it
                            # But don't deselect objects that were selected at the start
                            if was_in_prev and not is_in_current and obj.name not in self._initial_selected:
                                try:
                                    obj.select_set(False)
                                except Exception:
                                    pass
                    
                    # Performance timing for live selection
                    t_start = time.perf_counter()
                    t_coord_conv_total = 0.0
                    t_metric_total = 0.0
                    t_matching_total = 0.0
                    t_selection_total = 0.0
                    objects_processed = 0
                    objects_in_box = 0
                    matching_calls = 0
                    selection_calls = 0
                    cache_hits = 0
                    cache_misses = 0
                    
                    # Perform smart selection on objects currently in box
                    # Valid object types
                    valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
                    # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
                    view_layer_objects = context.view_layer.objects
                    
                    # Use cached metrics if available (major performance optimization)
                    use_cache = hasattr(self, '_object_metrics_cache') and self._object_metrics_cache is not None
                    
                    for obj in view_layer_objects:
                        objects_processed += 1
                        if obj.type not in valid_types:
                            continue
                        if obj.hide_viewport:
                            continue
                        
                        t_coord_start = time.perf_counter()
                        co_world = obj.matrix_world.translation
                        co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                        t_coord_conv_total += time.perf_counter() - t_coord_start
                        
                        if co_2d is None:
                            continue
                        x, y = co_2d
                        if xmin <= x <= xmax and ymin <= y <= ymax:
                            objects_in_box += 1
                            
                            # Use cached metric if available, otherwise calculate (fallback)
                            t_metric_start = time.perf_counter()
                            if use_cache and obj.name in self._object_metrics_cache:
                                obj_type, obj_metric = self._object_metrics_cache[obj.name]
                                cache_hits += 1
                            else:
                                # Fallback: calculate if cache not available
                                obj_type, obj_metric = get_object_metric(obj, self.depsgraph, self.use_evaluated)
                                cache_misses += 1
                            t_metric_total += time.perf_counter() - t_metric_start
                            
                            # Check if Ctrl is held - select all objects of same type regardless of parameters
                            if self._ctrl_pressed:
                                # Check if this object type matches any of the reference object types
                                # Use cached ref types set instead of list comprehension
                                if not hasattr(self, '_ref_types_set') or self._ref_types_set is None:
                                    self._ref_types_set = {r_type for r_type, _ in self._ref_metrics}
                                if obj_type in self._ref_types_set:
                                    t_sel_start = time.perf_counter()
                                    try:
                                        obj.select_set(True)
                                        selection_calls += 1
                                    except Exception:
                                        pass
                                    t_selection_total += time.perf_counter() - t_sel_start
                                continue
                            
                            # For lights, use specialized matching function that checks light type FIRST
                            if obj_type == 'LIGHT' and not self._ctrl_pressed:
                                # Use specialized function that checks light type before power
                                t_match_start = time.perf_counter()
                                matches = _matches_light_with_type_check(obj, obj_type, obj_metric, self._ref_objects, self._ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                matching_calls += 1
                                t_matching_total += time.perf_counter() - t_match_start
                            elif obj_type == 'FONT' and not self._ctrl_pressed:
                                # Use specialized function that checks text content before metric
                                t_match_start = time.perf_counter()
                                matches = _matches_text_with_content_check(obj, obj_type, obj_metric, self._ref_objects, self._ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                matching_calls += 1
                                t_matching_total += time.perf_counter() - t_match_start
                            else:
                                # For non-lights, non-text, or Ctrl mode, use normal filtering
                                if not self.include_other_types:
                                    # Use pre-computed filtered cache instead of list comprehension
                                    if not hasattr(self, '_ref_metrics_by_type') or self._ref_metrics_by_type is None:
                                        # Fallback: build cache on first use
                                        self._ref_metrics_by_type = {}
                                        for r_type, r_metric in self._ref_metrics:
                                            if r_type not in self._ref_metrics_by_type:
                                                self._ref_metrics_by_type[r_type] = []
                                            self._ref_metrics_by_type[r_type].append((r_type, r_metric))
                                    refs_to_check = self._ref_metrics_by_type.get(obj_type, [])
                                    if not refs_to_check:
                                        continue
                                    t_match_start = time.perf_counter()
                                    matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                    matching_calls += 1
                                    t_matching_total += time.perf_counter() - t_match_start
                                else:
                                    # Use cached list instead of creating new one
                                    if not hasattr(self, '_ref_metrics_list_cache') or self._ref_metrics_list_cache is None:
                                        self._ref_metrics_list_cache = list(self._ref_metrics)
                                    refs_to_check = self._ref_metrics_list_cache
                                    t_match_start = time.perf_counter()
                                    matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                    matching_calls += 1
                                    t_matching_total += time.perf_counter() - t_match_start
                            
                            t_sel_start = time.perf_counter()
                            try:
                                obj.select_set(matches)
                                selection_calls += 1
                            except Exception:
                                pass
                            t_selection_total += time.perf_counter() - t_sel_start
                    
                    t_total = time.perf_counter() - t_start
                    t_unaccounted = t_total - t_coord_conv_total - t_metric_total - t_matching_total - t_selection_total
                    
                    # Report performance metrics
                    print(f"[B Selection - LIVE SELECT] Total: {t_total*1000:.2f}ms | "
                          f"Objects: {objects_processed} processed, {objects_in_box} in box | "
                          f"Coord conv: {t_coord_conv_total*1000:.2f}ms | "
                          f"Get metrics: {t_metric_total*1000:.2f}ms ({cache_hits} cache hits, {cache_misses} misses) | "
                          f"Matching: {t_matching_total*1000:.2f}ms ({matching_calls} calls) | "
                          f"Selection: {t_selection_total*1000:.2f}ms ({selection_calls} calls) | "
                          f"Unaccounted: {t_unaccounted*1000:.2f}ms")
                    
                    # Store current box bounds for next frame
                    self._prev_box_bounds = (xmin, xmax, ymin, ymax)
                
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # finish on left release (selection)
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self.drawing and not self._deselect_mode:
            # finalize selection box
            x1, y1 = self.start
            x2, y2 = self.end
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)

            region = context.region
            rv3d = context.region_data

            candidates = []
            # Valid object types
            valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
            # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
            view_layer_objects = context.view_layer.objects
            
            for obj in view_layer_objects:
                if obj.type not in valid_types:
                    continue
                if obj.hide_viewport:
                    continue
                co_world = obj.matrix_world.translation
                co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                if co_2d is None:
                    continue
                x, y = co_2d
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    candidates.append(obj)

            # NOTE: Removed deselect-all so selection is additive across repeated uses.
            matched = []
            for obj in candidates:
                obj_type, obj_metric = get_object_metric(obj, self.depsgraph, self.use_evaluated)

                # Check if Ctrl is held - select all objects of same type regardless of parameters
                if self._ctrl_pressed:
                    # Check if this object type matches any of the reference object types
                    ref_types = [r_type for r_type, _ in self._ref_metrics]
                    if obj_type in ref_types:
                        try:
                            obj.select_set(True)
                        except Exception:
                            pass
                        matched.append(obj.name)
                    continue

                if not self.include_other_types:
                    # only consider refs with matching type
                    refs_to_check = [(r_type, r_metric) for r_type, r_metric in self._ref_metrics if r_type == obj_type]
                    if not refs_to_check:
                        continue
                else:
                    refs_to_check = list(self._ref_metrics)

                # For lights, use specialized matching function that checks light type FIRST
                if obj_type == 'LIGHT' and not self._ctrl_pressed:
                    # Use specialized function that checks light type before power
                    matches = _matches_light_with_type_check(obj, obj_type, obj_metric, self._ref_objects, self._ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                else:
                    # For non-lights or Ctrl mode, use normal matching
                    matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)

                if matches:
                    try:
                        obj.select_set(True)
                    except Exception:
                        pass
                    matched.append(obj.name)

            # cleanup handler
            if self._handler is not None:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(self._handler, 'WINDOW')
                except Exception:
                    pass
                self._handler = None

            # Remove status bar handler
            if self._statusbar_handler is not None:
                try:
                    bpy.types.STATUSBAR_HT_header.remove(self._statusbar_handler)
                except:
                    pass
                self._statusbar_handler = None
            
            # Stop animation timer if active
            self.animated_zebra_mode = False  # This will cause the timer to stop itself
            _last_custom_modal = None
            context.area.tag_redraw()
            return {'FINISHED'}

        return {'RUNNING_MODAL'}


def _build_dashed_segments(x0, y0, x1, y1, dash=DASH_PIXELS, gap=GAP_PIXELS):
    """Return list of line segment endpoint pairs for a dashed line between (x0,y0)-(x1,y1)."""
    segments = []
    if x0 == x1:  # vertical
        length = abs(y1 - y0)
        start_y = min(y0, y1)
        pos = 0
        while pos < length:
            s = start_y + pos
            e = start_y + min(pos + dash, length)
            # Always use the same x coordinate and calculate y from start
            segments.append((x0, s, x1, e))
            pos += dash + gap
    elif y0 == y1:  # horizontal
        length = abs(x1 - x0)
        start_x = min(x0, x1)
        pos = 0
        while pos < length:
            s = start_x + pos
            e = start_x + min(pos + dash, length)
            # Always use the same y coordinate and calculate x from start
            segments.append((s, y0, e, y1))
            pos += dash + gap
    return segments


def _build_dashed_segments_animated(x0, y0, x1, y1, dash=DASH_PIXELS, gap=GAP_PIXELS, offset=0.0, travel_cycles=1.0):
    """Return list of line segment endpoint pairs for an animated dashed line between (x0,y0)-(x1,y1) with animation offset.
    The offset creates a moving animation effect as it increases over time. This version ensures seamless looping.
    travel_cycles controls how many pattern cycles to travel before wrapping (affects wrap distance, not speed)."""
    segments = []
    pattern_length = dash + gap
    if pattern_length <= 0:
        return segments
    
    # Calculate effective wrap length (pattern repeats at pattern_length, but wraps at travel_cycles * pattern_length)
    effective_wrap_length = pattern_length * travel_cycles if travel_cycles > 0 else pattern_length
    
    if x0 == x1:  # vertical
        length = abs(y1 - y0)
        start_y = min(y0, y1)
        # Normalize offset to effective wrap length for seamless looping
        # This allows longer travel before wrap without affecting animation speed
        # The key: normalize by effective_wrap_length so offset wraps at travel_cycles * pattern_length
        if effective_wrap_length > 0:
            # Use modulo to normalize offset - this controls when the pattern wraps
            normalized_offset = offset % effective_wrap_length
            if normalized_offset < 0:
                normalized_offset += effective_wrap_length
        else:
            normalized_offset = 0
        
        # Start position: begin slightly before 0 to account for offset, ensuring seamless wrap
        # The offset controls where the pattern starts - higher travel_cycles = longer before wrap
        pos = -normalized_offset
        # Adjust pos to be positive, using effective_wrap_length to maintain correct wrap distance
        if pos < 0:
            # Use modulo with effective_wrap_length to maintain correct wrap behavior
            pos = pos % effective_wrap_length
            if pos < 0:
                pos += effective_wrap_length
        
        # Continue until we've covered the entire line length plus one pattern to ensure seamless wrap
        end_y = start_y + length
        while pos < length + pattern_length:
            s = start_y + pos
            e = start_y + min(pos + dash, length)
            # Allow segments that are within bounds, including those that end exactly at the corner
            # Ensure segments can extend all the way to the corner for complete coverage
            if e > s:
                # Clamp start to valid range [start_y, end_y], but allow end to reach exactly to end_y (corner)
                s_clamped = max(start_y, min(s, end_y))
                e_clamped = min(e, end_y)
                # Add segment if it has non-zero length and is within bounds
                # Allow segments that start anywhere from start_y up to (but not including) end_y
                if e_clamped > s_clamped and s_clamped >= start_y:
                    # Always use the same x coordinate and calculate y from start
                    segments.append((x0, s_clamped, x1, e_clamped))
            pos += pattern_length
    elif y0 == y1:  # horizontal
        length = abs(x1 - x0)
        start_x = min(x0, x1)
        # Normalize offset to effective wrap length for seamless looping
        # This allows longer travel before wrap without affecting animation speed
        # The key: normalize by effective_wrap_length so offset wraps at travel_cycles * pattern_length
        if effective_wrap_length > 0:
            # Use modulo to normalize offset - this controls when the pattern wraps
            normalized_offset = offset % effective_wrap_length
            if normalized_offset < 0:
                normalized_offset += effective_wrap_length
        else:
            normalized_offset = 0
        
        # Start position: begin slightly before 0 to account for offset, ensuring seamless wrap
        # The offset controls where the pattern starts - higher travel_cycles = longer before wrap
        pos = -normalized_offset
        # Adjust pos to be positive, using effective_wrap_length to maintain correct wrap distance
        if pos < 0:
            # Use modulo with effective_wrap_length to maintain correct wrap behavior
            pos = pos % effective_wrap_length
            if pos < 0:
                pos += effective_wrap_length
        
        # Continue until we've covered the entire line length plus one pattern to ensure seamless wrap
        end_x = start_x + length
        while pos < length + pattern_length:
            s = start_x + pos
            e = start_x + min(pos + dash, length)
            # Allow segments that are within bounds, including those that end exactly at the corner
            # Ensure segments can extend all the way to the corner for complete coverage
            if e > s:
                # Clamp start to valid range [start_x, end_x], but allow end to reach exactly to end_x (corner)
                s_clamped = max(start_x, min(s, end_x))
                e_clamped = min(e, end_x)
                # Add segment if it has non-zero length and is within bounds
                # Allow segments that start anywhere from start_x up to (but not including) end_x
                if e_clamped > s_clamped and s_clamped >= start_x:
                    # Always use the same y coordinate and calculate x from start
                    segments.append((s_clamped, y0, e_clamped, y1))
            pos += pattern_length
    return segments


def _build_dashed_segments_animated_circle(cx, cy, radius, dash, gap, offset=0.0, travel_cycles=1.0, num_segments=256):
    """Return list of line segment vertices for an animated dashed circle with animation offset.
    The offset creates a moving animation effect as it increases over time.
    Returns a list of vertex lists, where each vertex list represents one dash segment."""
    segments = []
    pattern_length = dash + gap
    if pattern_length <= 0 or radius <= 0:
        return segments
    
    circumference = 2 * math.pi * radius
    if circumference <= 0:
        return segments
    
    # Calculate effective wrap length
    effective_wrap_length = pattern_length * travel_cycles if travel_cycles > 0 else pattern_length
    
    # Normalize offset to effective wrap length
    if effective_wrap_length > 0:
        normalized_offset = offset % effective_wrap_length
        if normalized_offset < 0:
            normalized_offset += effective_wrap_length
    else:
        normalized_offset = 0
    
    # Calculate angle per pixel
    angle_per_pixel = (2 * math.pi) / circumference
    
    # Start position accounting for offset
    pos = -normalized_offset
    if pos < 0:
        pos = pos % effective_wrap_length
        if pos < 0:
            pos += effective_wrap_length
    
    # Generate dashes around the circle
    while pos < circumference + pattern_length:
        dash_end = min(pos + dash, circumference)
        if dash_end > pos:
            start_angle = pos * angle_per_pixel
            end_angle = dash_end * angle_per_pixel
            
            # Create smooth segments for this dash
            dash_length_pixels = dash_end - pos
            # Increase minimum segments for better curve quality and anti-aliasing
            num_dash_segs = max(8, int((dash_length_pixels / circumference) * num_segments))
            dash_verts = []
            for j in range(num_dash_segs + 1):
                t = j / num_dash_segs if num_dash_segs > 0 else 0
                angle = start_angle + (end_angle - start_angle) * t
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                dash_verts.append((x, y))
            
            if len(dash_verts) >= 2:
                segments.append(dash_verts)
        
        pos += pattern_length
    
    return segments


class VIEW3D_OT_box_select_toggle(bpy.types.Operator):
    """Press B: first press -> enter waiting-mode (zebra crosshair). Press B again -> cancel waiting-mode and run the custom similar-verts box select."""
    bl_idname = "view3d.box_select_toggle"
    bl_label = "Box Select Indefinite Toggle (Zebra)"
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}

    def draw_crosshair(self, context):
        # Safety check: verify operator instance is still valid
        try:
            # Try to access a simple attribute to check if operator is still alive
            _ = self.bl_idname
        except (ReferenceError, AttributeError):
            # Operator has been removed, don't draw anything
            return
        
        # Check if we should cancel
        if getattr(self, "_should_cancel", False):
            return
        
        try:
            # Draw zebra crosshair and (when drawing) the selection box
            region = context.region
            
            # Get current_mouse safely
            current_mouse = _safe_getattr(self, "current_mouse", None)
            drawing = _safe_getattr(self, 'drawing', False)

            if current_mouse is not None and not drawing:
                x, y = current_mouse

                try:
                    gpu.state.blend_set('ALPHA')
                except Exception:
                    pass

                shader = gpu.shader.from_builtin('UNIFORM_COLOR')

                # Vertical dashed segments
                # Get dash settings from preferences
                prefs = _get_addon_preferences()
                dash_pixels = prefs.zebra_dash_pixels if prefs else DASH_PIXELS
                gap_pixels = prefs.zebra_gap_pixels if prefs else GAP_PIXELS
                v_segments = _build_dashed_segments(x, 0, x, region.height, dash_pixels, gap_pixels)
                # Split alternating segments into two lists to draw with alternating colors (zebra)
                v_even = []
                v_odd = []
                for i, seg in enumerate(v_segments):
                    x0, y0, x1, y1 = seg
                    if i % 2 == 0:
                        v_even.extend([(x0, y0), (x1, y1)])
                    else:
                        v_odd.extend([(x0, y0), (x1, y1)])

                # Horizontal dashed segments
                # Get dash settings from preferences
                prefs = _get_addon_preferences()
                dash_pixels = prefs.zebra_dash_pixels if prefs else DASH_PIXELS
                gap_pixels = prefs.zebra_gap_pixels if prefs else GAP_PIXELS
                h_segments = _build_dashed_segments(0, y, region.width, y, dash_pixels, gap_pixels)
                h_even = []
                h_odd = []
                for i, seg in enumerate(h_segments):
                    x0, y0, x1, y1 = seg
                    if i % 2 == 0:
                        h_even.extend([(x0, y0), (x1, y1)])
                    else:
                        h_odd.extend([(x0, y0), (x1, y1)])

                # Get outline color from preferences
                outline_color = tuple(prefs.selection_outline_color) if prefs else (1.0, 1.0, 1.0, 1.0)
                # Ensure 4 components (RGBA) - handle old 3-component colors
                if len(outline_color) == 3:
                    outline_color = outline_color + (1.0,)
                
                # Draw even segments (bright stripe)
                if v_even or h_even:
                    batch = batch_for_shader(shader, 'LINES', {"pos": (v_even + h_even)}) if (v_even + h_even) else None
                    if batch:
                        shader.bind()
                        color_alpha = outline_color[3] if len(outline_color) > 3 else 1.0
                        shader.uniform_float("color", (outline_color[0], outline_color[1], outline_color[2], 0.9 * color_alpha))
                        batch.draw(shader)

                # Draw odd segments (dark stripe - using darker version of outline color)
                if v_odd or h_odd:
                    batch = batch_for_shader(shader, 'LINES', {"pos": (v_odd + h_odd)}) if (v_odd + h_odd) else None
                    if batch:
                        shader.bind()
                        # Use darker version of outline color for zebra effect
                        dark_factor = 0.3  # Darken by 70%
                        color_alpha = outline_color[3] if len(outline_color) > 3 else 1.0
                        shader.uniform_float("color", (outline_color[0] * dark_factor, outline_color[1] * dark_factor, outline_color[2] * dark_factor, 0.5 * color_alpha))
                        batch.draw(shader)

                try:
                    gpu.state.blend_set('NONE')
                except Exception:
                    pass

            # Draw selection box when drawing (with dashed/punctured outline in waiting mode)
            # Only draw box when actively drawing (not just when start/end exist)
            if getattr(self, 'drawing', False) and hasattr(self, "start") and hasattr(self, "end") and self.start is not None and self.end is not None:
                x1, y1 = self.start
                x2, y2 = self.end
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                verts = [
                    (float(xmin), float(ymin)),
                    (float(xmax), float(ymin)),
                    (float(xmax), float(ymax)),
                    (float(xmin), float(ymax)),
                ]
                try:
                    gpu.state.blend_set('ALPHA')
                except Exception:
                    pass
                shader = gpu.shader.from_builtin('UNIFORM_COLOR')
                batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
                shader.bind()
                shader.uniform_float("color", (1.0, 1.0, 1.0, 0.02))
                batch.draw(shader)
                
                # Draw dashed/punctured box outline
                try:
                    gpu.state.line_width_set(1.5)  # Slightly thicker for visibility
                except Exception:
                    pass
                
                # Build dashed segments for all four sides of the box
                # Get dash settings from preferences
                prefs = _get_addon_preferences()
                box_dash_len = prefs.box_dash_length if prefs else BOX_DASH_LENGTH
                box_gap_len = prefs.box_gap_length if prefs else BOX_GAP_LENGTH
                outline_color = tuple(prefs.selection_outline_color) if prefs else (1.0, 1.0, 1.0, 1.0)
                # Ensure 4 components (RGBA) - handle old 3-component colors
                if len(outline_color) == 3:
                    outline_color = outline_color + (1.0,)
                
                box_dash_segments = []
                # Top edge (left to right)
                top_segs = _build_dashed_segments(xmin, ymax, xmax, ymax, box_dash_len, box_gap_len)
                for seg in top_segs:
                    x0, y0, x1, y1 = seg
                    box_dash_segments.extend([(x0, y0), (x1, y1)])
                # Right edge (top to bottom)
                right_segs = _build_dashed_segments(xmax, ymax, xmax, ymin, box_dash_len, box_gap_len)
                for seg in right_segs:
                    x0, y0, x1, y1 = seg
                    box_dash_segments.extend([(x0, y0), (x1, y1)])
                # Bottom edge (right to left)
                bottom_segs = _build_dashed_segments(xmax, ymin, xmin, ymin, box_dash_len, box_gap_len)
                for seg in bottom_segs:
                    x0, y0, x1, y1 = seg
                    box_dash_segments.extend([(x0, y0), (x1, y1)])
                # Left edge (bottom to top)
                left_segs = _build_dashed_segments(xmin, ymin, xmin, ymax, box_dash_len, box_gap_len)
                for seg in left_segs:
                    x0, y0, x1, y1 = seg
                    box_dash_segments.extend([(x0, y0), (x1, y1)])
                
                # Draw dashed box outline
                if box_dash_segments:
                    batch = batch_for_shader(shader, 'LINES', {"pos": box_dash_segments})
                    shader.bind()
                    color_alpha = outline_color[3] if len(outline_color) > 3 else 1.0
                    shader.uniform_float("color", (outline_color[0], outline_color[1], outline_color[2], 1.0 * color_alpha))
                    batch.draw(shader)
                
                try:
                    gpu.state.blend_set('NONE')
                except Exception:
                    pass
        except (ReferenceError, AttributeError):
            # Operator instance has been removed, silently return
            return

    def invoke(self, context, event):
        global _last_toggle_modal, _last_custom_modal

        # Only work in OBJECT mode - otherwise pass through to Blender's default box select
        if context.mode != 'OBJECT':
            override = _find_3dview_override()
            if override is not None:
                try:
                    bpy.ops.view3d.select_box(override, 'INVOKE_DEFAULT')
                except Exception:
                    try:
                        bpy.ops.view3d.select_box('INVOKE_DEFAULT')
                    except Exception:
                        pass
            else:
                try:
                    bpy.ops.view3d.select_box('INVOKE_DEFAULT')
                except Exception:
                    pass
            return {'CANCELLED'}

        # --- NEW: if user is editing a curve, do nothing special — call Blender's builtin and exit ---
        if _is_curve_edit_mode(context):
            override = _find_3dview_override()
            if override is not None:
                try:
                    bpy.ops.view3d.select_box(override, 'INVOKE_DEFAULT')
                except Exception:
                    try:
                        bpy.ops.view3d.select_box('INVOKE_DEFAULT')
                    except Exception:
                        pass
            else:
                try:
                    bpy.ops.view3d.select_box('INVOKE_DEFAULT')
                except Exception:
                    pass
            return {'CANCELLED'}
        # --- end NEW check ---

        # If there is an existing waiting modal (from prior B) and it's not this instance,
        # that means user pressed B again — cancel the waiting modal and run the custom operator.
        if _last_toggle_modal is not None and _last_toggle_modal is not self:
            # Cancel the previous modal
            try:
                _last_toggle_modal._should_cancel = True
                if hasattr(_last_toggle_modal, "_handler") and _last_toggle_modal._handler is not None:
                    try:
                        bpy.types.SpaceView3D.draw_handler_remove(_last_toggle_modal._handler, 'WINDOW')
                    except Exception:
                        pass
                    _last_toggle_modal._handler = None
            except (ReferenceError, AttributeError, Exception):
                pass
            _last_toggle_modal = None
            # Force redraw to clear any ghost crosshairs
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            # directly invoke the custom similar-verts operator
            try:
                return bpy.ops.view3d.select_box_similar_verts('INVOKE_DEFAULT')
            except Exception:
                # fallback - just cancel
                return {'CANCELLED'}

        # Otherwise, first press -> enter waiting-mode indefinitely
        # If a custom modal is active, cancel it and go into waiting-mode
        if _last_custom_modal is not None:
            try:
                _last_custom_modal._should_cancel = True
                if hasattr(_last_custom_modal, "_handler") and _last_custom_modal._handler is not None:
                    try:
                        bpy.types.SpaceView3D.draw_handler_remove(_last_custom_modal._handler, 'WINDOW')
                    except Exception:
                        pass
                    _last_custom_modal._handler = None
            except Exception:
                pass
            _last_custom_modal = None

        # Initialize state before registering handler
        self.current_mouse = (event.mouse_region_x, event.mouse_region_y)
        self.start_time = None
        self._should_cancel = False
        self.start = None
        self.end = None
        self.drawing = False
        self._deselect_mode = False
        
        # Ensure we don't have a leftover handler from a previous instance
        if hasattr(self, "_handler") and self._handler is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handler, 'WINDOW')
            except Exception:
                pass
            self._handler = None
        
        # Register the draw handler for crosshair
        self._handler = bpy.types.SpaceView3D.draw_handler_add(self.draw_crosshair, (context,), 'WINDOW', 'POST_PIXEL')

        # remember this modal instance so a second B can cancel it
        _last_toggle_modal = self

        context.window_manager.modal_handler_add(self)
        # Force redraw to show crosshair immediately
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        global _last_toggle_modal

        # Update mouse position first (before checking for cancel)
        # This ensures crosshair follows mouse even if about to be canceled
        if event.type == 'MOUSEMOVE':
            self.current_mouse = (event.mouse_region_x, event.mouse_region_y)
            if self.drawing:
                self.end = self.current_mouse
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # If a second B (handled in invoke) flagged this modal to cancel, exit quietly
        if getattr(self, "_should_cancel", False):
            if hasattr(self, "_handler") and self._handler is not None:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(self._handler, 'WINDOW')
                except Exception:
                    pass
                self._handler = None
            if _last_toggle_modal is self:
                _last_toggle_modal = None
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            return {'CANCELLED'}

        # If user presses B again while this modal is active, Blender will call invoke on a new instance
        # which will cancel this modal and start the custom operator. So just pass through.
        # Don't do cleanup here - let invoke handle it to prevent race conditions
        if event.type == 'B' and event.value == 'PRESS':
            return {'PASS_THROUGH'}

        # start drawing on left press (we implement the drag/select here so builtin gets not lost)
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and not self.drawing:
            self.start = (event.mouse_region_x, event.mouse_region_y)
            self.end = self.start
            self.current_mouse = self.start
            self.drawing = True
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # finish drawing and perform ADD-selection on left release
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self.drawing:
            # finalize selection rect
            x1, y1 = self.start
            x2, y2 = self.end
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)

            region = context.region
            rv3d = context.region_data

            # Valid object types
            valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
            # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
            view_layer_objects = context.view_layer.objects

            for obj in view_layer_objects:
                if obj.type not in valid_types:
                    continue
                if obj.hide_viewport:
                    continue
                co_world = obj.matrix_world.translation
                co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                if co_2d is None:
                    continue
                x, y = co_2d
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    try:
                        obj.select_set(True)  # ADD behaviour
                    except Exception:
                        pass

            # cleanup handler
            if self._handler is not None:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(self._handler, 'WINDOW')
                except Exception:
                    pass
                self._handler = None

            _last_toggle_modal = None
            context.area.tag_redraw()
            return {'FINISHED'}

        # If user clicks other keys, cancel waiting and call builtin box select (fallback)
        if event.type in {'RIGHTMOUSE', 'ESC', 'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            if self._handler is not None:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(self._handler, 'WINDOW')
                except Exception:
                    pass
                self._handler = None
            _last_toggle_modal = None
            context.area.tag_redraw()

            # fallback to builtin
            override = _find_3dview_override()
            if override is not None:
                try:
                    bpy.ops.view3d.select_box(override, 'INVOKE_DEFAULT', mode='ADD')
                except Exception:
                    try:
                        bpy.ops.view3d.select_box(override, 'INVOKE_DEFAULT')
                    except Exception:
                        pass
            else:
                try:
                    bpy.ops.view3d.select_box('INVOKE_DEFAULT', mode='ADD')
                except Exception:
                    try:
                        bpy.ops.view3d.select_box('INVOKE_DEFAULT')
                    except Exception:
                        pass
            return {'FINISHED'}

        # Middle mouse: when waiting-mode (single B) we cancel our modal and invoke Blender's built-in
        # box select in SUB (deselect) mode so user gets the usual deselect behavior.
        if event.type == 'MIDDLEMOUSE' and event.value == 'PRESS':
            # cleanup handler
            if self._handler is not None:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(self._handler, 'WINDOW')
                except Exception:
                    pass
                self._handler = None
            _last_toggle_modal = None
            context.area.tag_redraw()

            # invoke builtin box select in subtract mode (best-effort; fall back to default if 'mode' unsupported)
            override = _find_3dview_override()
            if override is not None:
                try:
                    bpy.ops.view3d.select_box(override, 'INVOKE_DEFAULT', mode='SUB')
                except Exception:
                    try:
                        bpy.ops.view3d.select_box(override, 'INVOKE_DEFAULT')
                    except Exception:
                        try:
                            bpy.ops.view3d.select_box('INVOKE_DEFAULT', mode='SUB')
                        except Exception:
                            try:
                                bpy.ops.view3d.select_box('INVOKE_DEFAULT')
                            except Exception:
                                pass
            else:
                try:
                    bpy.ops.view3d.select_box('INVOKE_DEFAULT', mode='SUB')
                except Exception:
                    try:
                        bpy.ops.view3d.select_box('INVOKE_DEFAULT')
                    except Exception:
                        pass

            return {'FINISHED'}

        return {'RUNNING_MODAL'}


class VIEW3D_OT_circle_select_similar_verts(bpy.types.Operator):
    """Press hotkey to arm; left-click and drag to select similar objects within circle radius."""
    bl_idname = "view3d.select_circle_similar_verts"
    bl_label = "Circle Select: Similar Vertex Count"
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}

    tolerance: IntProperty(
        name="Absolute tolerance",
        default=0,
        min=0,
        description="Allowed absolute difference in object metric (if Use Percentage is off). Default 0 = exact match only."
    )
    use_percentage: BoolProperty(
        name="Use Percentage",
        default=False,
        description="Interpret tolerance as percentage of reference object's metric"
    )
    percent_tolerance: FloatProperty(
        name="Percent tolerance",
        default=0.0,
        min=0.0,
        max=100.0,
        description="Allowed percentage difference (if Use Percentage on). Default 0.0 = exact match only."
    )
    use_evaluated: BoolProperty(
        name="Use evaluated data (modifiers)",
        default=False,
        description="Count vertices after modifiers are applied (may be slower for meshes)"
    )
    include_other_types: BoolProperty(
        name="Include other object types",
        default=False,
        description="Allow matching objects of different Blender types (e.g. match Empties to Meshes) using the computed metric"
    )
    match_all_selected: BoolProperty(
        name="Match to all selected",
        default=True,
        description="Match candidates against ANY of the currently selected objects (instead of only the active object)"
    )

    def draw_callback(self, context):
        # Safety check: verify operator instance is still valid
        try:
            # Try to access a simple attribute to check if operator is still alive
            _ = self.bl_idname
        except (ReferenceError, AttributeError):
            # Operator has been removed, don't draw anything
            return
        
        try:
            region = context.region

            # Draw circle at mouse position when armed (before drawing starts)
            # Only draw when NOT drawing (before first click or after release)
            current_mouse = _safe_getattr(self, "current_mouse", None)
            drawing = _safe_getattr(self, 'drawing', False)
            
            if current_mouse is not None and not drawing:
                cx, cy = current_mouse
                # Use current radius (can be changed with mouse wheel)
                radius = _safe_getattr(self, 'radius', 25.0)
                
                # Draw circle at mouse position - dashed or solid based on style
                num_segments = 256  # Increased for smoother appearance
                use_dashed = _safe_getattr(self, 'use_dashed', True)
                
                # Create circle vertices
                circle_verts = []
                for i in range(num_segments + 1):
                    angle = (i / num_segments) * 2 * math.pi
                    x = cx + radius * math.cos(angle)
                    y = cy + radius * math.sin(angle)
                    circle_verts.append((x, y))
                
                try:
                    gpu.state.blend_set('ALPHA')
                except Exception:
                    pass

                shader = gpu.shader.from_builtin('UNIFORM_COLOR')
                try:
                    gpu.state.line_width_set(1.5)  # Slightly thicker line for smoother appearance
                except Exception:
                    pass
                
                # Check if animated zebra mode is active
                if _safe_getattr(self, 'animated_zebra_mode', False):
                    # Draw animated zebra dashes with glow effect
                    prefs = _get_addon_preferences()
                    anim_dash_len = prefs.animated_zebra_dash_length if prefs else 10.0
                    anim_gap_len = prefs.animated_zebra_gap_length if prefs else 10.0
                    animation_speed = prefs.animated_zebra_speed if prefs else 18.0
                    travel_cycles = 1.0  # Fixed to 1.0
                    glow_intensity_base = 0.7
                    
                    # Calculate animation offset based on time for smooth continuous movement
                    current_time = time.time()
                    continuous_offset = current_time * animation_speed * 2
                    
                    # Calculate circumference for offset calculation
                    circumference = 2 * math.pi * radius
                    if circumference > 0:
                        # Calculate local offset for circle (clockwise motion)
                        local_offset = -continuous_offset
                        
                        # Build animated dashed segments for circle
                        circle_dash_segments = _build_dashed_segments_animated_circle(
                            cx, cy, radius, anim_dash_len, anim_gap_len, local_offset, travel_cycles, num_segments
                        )
                        
                        # Get outline color from preferences
                        outline_color = tuple(prefs.selection_outline_color) if prefs else (1.0, 1.0, 1.0, 1.0)
                        if len(outline_color) == 3:
                            outline_color = outline_color + (1.0,)
                        
                        # Calculate pulsing glow intensity
                        glow_pulse_speed = 2.0  # cycles per second
                        glow_pulse = (math.sin(current_time * glow_pulse_speed * 2 * math.pi) + 1.0) / 2.0  # 0.0 to 1.0
                        glow_intensity = 0.3 + (glow_pulse * 0.7 * glow_intensity_base)
                        
                        color_alpha = outline_color[3] if len(outline_color) > 3 else 1.0
                        
                        # Draw gradient glow passes
                        glow_passes = [
                            (3.0, 0.12),
                            (2.5, 0.20),
                            (2.0, 0.30),
                            (1.5, 0.45),
                        ]
                        
                        for line_width, base_glow_alpha in glow_passes:
                            try:
                                gpu.state.line_width_set(line_width * glow_intensity)
                            except Exception:
                                pass
                            
                            for dash_verts in circle_dash_segments:
                                if len(dash_verts) >= 2:
                                    batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": dash_verts})
                                    shader.bind()
                                    final_alpha = base_glow_alpha * glow_intensity * color_alpha
                                    shader.uniform_float("color", (outline_color[0], outline_color[1], outline_color[2], final_alpha))
                                    batch.draw(shader)
                        
                        # Draw main dashes with anti-aliasing simulation (multiple passes with slight offsets)
                        # Draw multiple passes with reduced opacity for smoother appearance
                        offsets = [(0, 0), (0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]
                        for offset_x, offset_y in offsets:
                            try:
                                gpu.state.line_width_set(1.5)
                            except Exception:
                                pass
                            for dash_verts in circle_dash_segments:
                                if len(dash_verts) >= 2:
                                    # Apply offset to vertices for anti-aliasing
                                    offset_verts = [(x + offset_x, y + offset_y) for x, y in dash_verts]
                                    batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": offset_verts})
                                    shader.bind()
                                    alpha = 0.3 if offset_x != 0 or offset_y != 0 else 0.5
                                    final_alpha = alpha * color_alpha
                                    shader.uniform_float("color", (outline_color[0], outline_color[1], outline_color[2], final_alpha))
                                    batch.draw(shader)
                elif use_dashed:
                    # Draw dashed pattern
                    # Get dash settings from preferences
                    prefs = _get_addon_preferences()
                    dash_length = prefs.circle_dash_length if prefs else CIRCLE_DASH_LENGTH
                    gap_length = prefs.circle_gap_length if prefs else CIRCLE_GAP_LENGTH
                    dash_list = []
                    
                    # Calculate circumference and create dash pattern
                    circumference = 2 * math.pi * radius
                    if circumference > 0:
                        pattern_length = dash_length + gap_length
                        angle_per_pixel = (2 * math.pi) / circumference
                        
                        pos = 0.0
                        while pos < circumference:
                            # Draw dash
                            dash_end = min(pos + dash_length, circumference)
                            start_angle = pos * angle_per_pixel
                            end_angle = dash_end * angle_per_pixel
                            
                            # Create smooth segments for this dash
                            num_dash_segs = max(3, int((dash_end - pos) / (circumference / num_segments)))
                            dash_verts = []
                            for j in range(num_dash_segs + 1):
                                t = j / num_dash_segs if num_dash_segs > 0 else 0
                                angle = start_angle + (end_angle - start_angle) * t
                                x = cx + radius * math.cos(angle)
                                y = cy + radius * math.sin(angle)
                                dash_verts.append((x, y))
                            
                            # Add this dash to the list
                            if len(dash_verts) >= 2:
                                dash_list.append(dash_verts)
                            
                            pos += pattern_length
                    
                    # Get outline color from preferences
                    prefs = _get_addon_preferences()
                    outline_color = tuple(prefs.selection_outline_color) if prefs else (1.0, 1.0, 1.0, 1.0)
                    # Ensure 4 components (RGBA) - handle old 3-component colors
                    if len(outline_color) == 3:
                        outline_color = outline_color + (1.0,)
                    
                    # Draw each dash separately
                    for dash_verts in dash_list:
                        if len(dash_verts) >= 2:
                            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": dash_verts})
                            shader.bind()
                            color_alpha = outline_color[3] if len(outline_color) > 3 else 1.0
                            shader.uniform_float("color", (outline_color[0], outline_color[1], outline_color[2], 0.8 * color_alpha))
                            batch.draw(shader)
                else:
                    # Get outline color from preferences
                    prefs = _get_addon_preferences()
                    outline_color = tuple(prefs.selection_outline_color) if prefs else (1.0, 1.0, 1.0, 1.0)
                    # Ensure 4 components (RGBA) - handle old 3-component colors
                    if len(outline_color) == 3:
                        outline_color = outline_color + (1.0,)
                    
                    # Draw solid line with anti-aliasing simulation (multiple passes with slight offsets)
                    # Draw multiple passes with reduced opacity for smoother appearance
                    offsets = [(0, 0), (0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]
                    for offset_x, offset_y in offsets:
                        offset_verts = [(x + offset_x, y + offset_y) for x, y in circle_verts]
                        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": offset_verts})
                        shader.bind()
                        alpha = 0.3 if offset_x != 0 or offset_y != 0 else 0.5
                        color_alpha = outline_color[3] if len(outline_color) > 3 else 1.0
                        shader.uniform_float("color", (outline_color[0], outline_color[1], outline_color[2], alpha * color_alpha))
                        batch.draw(shader)
                
                # Draw gradient visualization only in smart mode (solid line, not dashed)
                # Gradient should only show when using smart selection, not standard selection
                # Don't show gradient or percentage when Ctrl mode is active (select all mode doesn't use tolerance)
                if not use_dashed:
                    try:
                        tolerance_multiplier = _safe_getattr(self, 'tolerance_multiplier', 100.0)
                    except (ReferenceError, AttributeError):
                        tolerance_multiplier = 100.0
                    ctrl_pressed = _safe_getattr(self, '_ctrl_pressed', False)
                    prefs = _get_addon_preferences()
                    show_gradient = prefs.show_gradient if prefs else True
                    
                    if tolerance_multiplier != 100.0 and show_gradient and not ctrl_pressed:
                        _draw_gradient_ring(cx, cy, radius, tolerance_multiplier, num_segments)
                
                if not use_dashed:
                    # Draw tolerance multiplier text in center of circle when not 100% (only in smart mode - solid line)
                    try:
                        tolerance_multiplier = _safe_getattr(self, 'tolerance_multiplier', 100.0)
                    except (ReferenceError, AttributeError):
                        tolerance_multiplier = 100.0
                    ctrl_pressed = _safe_getattr(self, '_ctrl_pressed', False)
                    prefs = _get_addon_preferences()
                    show_percentage = prefs.show_percentage_text if prefs else True
                    
                    if tolerance_multiplier != 100.0 and show_percentage and not ctrl_pressed:
                        # Get text settings from preferences
                        base_font_size = prefs.text_font_size if prefs else 15
                        text_size_scale = prefs.text_size_scale if prefs else 0.35
                        text_color = prefs.text_color if prefs else (1.0, 1.0, 1.0, 1.0)
                        # Ensure 4 components (RGBA) - handle old 3-component colors
                        if len(text_color) == 3:
                            text_color = text_color + (1.0,)
                        text_color_r, text_color_g, text_color_b = text_color[0], text_color[1], text_color[2]
                        text_placement = prefs.text_placement_circle if prefs else 'MOUSE'
                        
                        # Calculate circle size for dynamic text scaling
                        # Use radius (diameter would be 2*radius, but radius is more intuitive)
                        circle_size = radius * 2.0  # Use diameter for scaling
                        
                        # Calculate dynamic font size: blend between fixed size and scaled size
                        # Scale factor: convert circle size to reasonable font size (e.g., 100px circle = ~15px font)
                        scale_factor = 0.15  # Adjust this to control scaling sensitivity
                        scaled_size = circle_size * scale_factor
                        text_font_size = base_font_size * (1.0 - text_size_scale) + scaled_size * text_size_scale
                        text_font_size = max(10, min(50, text_font_size))  # Clamp between 10 and 50
                        
                        # Calculate opacity based on circle size (fade out when too small)
                        # Minimum size where text is fully visible: around 40 pixels diameter (for readable text)
                        # Below this, fade from 0 (at 0px) to 1.0 (at 40px)
                        min_size_threshold = 40.0
                        if circle_size <= 0:
                            text_alpha = 0.0
                        elif circle_size >= min_size_threshold:
                            text_alpha = 1.0
                        else:
                            # Smooth fade: 0 to 1.0 as size goes from 0 to min_size_threshold
                            # Use a power curve for faster fade (make it disappear quickly when small)
                            normalized_size = circle_size / min_size_threshold
                            text_alpha = pow(normalized_size, 2.5)  # Power curve for faster fade
                        
                        # Don't draw if too transparent
                        if text_alpha < 0.01:
                            pass  # Skip drawing
                        else:
                            # Determine text position: cx, cy is mouse position before drawing, circle center after
                            text_center_x = cx  # Use mouse position if CENTER mode not yet available
                            text_center_y = cy
                            if text_placement == 'CENTER':
                                center_val = _safe_getattr(self, 'center', None)
                                if center_val is not None:
                                    # Use actual circle center if available
                                    text_center_x, text_center_y = center_val
                            
                            # Get display mode and format text accordingly
                            # If multiple object types are selected, always show percentage
                            force_percentage = _has_multiple_object_types(self)
                            text_display_mode = prefs.text_display_mode if prefs else 'PERCENTAGE'
                            if text_display_mode == 'VERTEX_COUNT' and not force_percentage:
                                try:
                                    threshold_value, obj_type = _get_vertex_count_threshold(self)
                                    if obj_type == 'LIGHT':
                                        text = _format_power_display(threshold_value) if threshold_value > 0 else "0W"
                                    else:
                                        text = _format_vertex_count_display(threshold_value) if threshold_value > 0 else "0"
                                except (ReferenceError, AttributeError):
                                    text = "0"
                            else:
                                text = _format_multiplier_display(tolerance_multiplier)
                            
                            font_id = 0
                            blf.size(font_id, int(text_font_size))
                            
                            # Get text dimensions first
                            try:
                                text_width, text_height = blf.dimensions(font_id, text)
                            except (AttributeError, TypeError):
                                # Fallback: approximate dimensions based on font size and character count
                                text_width = text_font_size * 0.6 * len(text)
                                text_height = text_font_size * 1.2
                            
                            # Calculate text position based on placement setting
                            if text_placement == 'CENTER':
                                # Center text: blf.position sets bottom-left, so we need to offset by half text dimensions
                                text_x = text_center_x - text_width / 2.0
                                text_y = text_center_y - text_height / 2.0
                            else:  # MOUSE
                                # Position at mouse cursor
                                text_x = cx
                                text_y = cy
                            
                            blf.position(font_id, text_x, text_y, 0)
                            # Multiply calculated alpha by color alpha (if available)
                            color_alpha = text_color[3] if len(text_color) > 3 else 1.0
                            final_text_alpha = text_alpha * color_alpha
                            blf.color(font_id, text_color_r, text_color_g, text_color_b, final_text_alpha)
                            blf.draw(font_id, text)

                    try:
                        gpu.state.blend_set('NONE')
                    except Exception:
                        pass

            # Draw selection circle if we're dragging
            center = _safe_getattr(self, "center", None)
            radius = _safe_getattr(self, "radius", None)
            
            if center is None or radius is None:
                return

            cx, cy = center

            # Draw circle - dashed or solid based on style
            num_segments = 256  # Increased for smoother appearance
            use_dashed = getattr(self, 'use_dashed', True)
            
            # Create circle vertices
            circle_verts = []
            for i in range(num_segments + 1):
                angle = (i / num_segments) * 2 * math.pi
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                circle_verts.append((x, y))

            try:
                gpu.state.blend_set('ALPHA')
            except Exception:
                pass

            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            
            # Draw filled circle (translucent)
            circle_verts = []
            for i in range(num_segments + 1):
                angle = (i / num_segments) * 2 * math.pi
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                circle_verts.append((x, y))
            
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": circle_verts})
            shader.bind()
            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.02))
            batch.draw(shader)

            # Draw circle outline - dashed or solid based on style
            try:
                gpu.state.line_width_set(1.5)  # Slightly thicker line for smoother appearance
            except Exception:
                pass
            
            # Check if animated zebra mode is active
            if getattr(self, 'animated_zebra_mode', False):
                # Draw animated zebra dashes with glow effect
                prefs = _get_addon_preferences()
                anim_dash_len = prefs.animated_zebra_dash_length if prefs else 10.0
                anim_gap_len = prefs.animated_zebra_gap_length if prefs else 10.0
                animation_speed = prefs.animated_zebra_speed if prefs else 18.0
                travel_cycles = 1.0  # Fixed to 1.0
                glow_intensity_base = 0.7
                gradient_dist = 2.0
                
                # Calculate animation offset based on time for smooth continuous movement
                current_time = time.time()
                continuous_offset = current_time * animation_speed * 2
                
                # Calculate circumference for offset calculation
                circumference = 2 * math.pi * radius
                if circumference > 0:
                    # Calculate local offset for circle (clockwise motion)
                    local_offset = -continuous_offset
                    
                    # Build animated dashed segments for circle
                    circle_dash_segments = _build_dashed_segments_animated_circle(
                        cx, cy, radius, anim_dash_len, anim_gap_len, local_offset, travel_cycles, num_segments
                    )
                    
                    # Get outline color from preferences
                    outline_color = tuple(prefs.selection_outline_color) if prefs else (1.0, 1.0, 1.0, 1.0)
                    if len(outline_color) == 3:
                        outline_color = outline_color + (1.0,)
                    
                    # Calculate pulsing glow intensity
                    glow_pulse_speed = 2.0  # cycles per second
                    glow_pulse = (math.sin(current_time * glow_pulse_speed * 2 * math.pi) + 1.0) / 2.0  # 0.0 to 1.0
                    glow_intensity = 0.3 + (glow_pulse * 0.7 * glow_intensity_base)
                    
                    color_alpha = outline_color[3] if len(outline_color) > 3 else 1.0
                    
                    # Draw gradient glow passes
                    glow_passes = [
                        (3.0, 0.12),
                        (2.5, 0.20),
                        (2.0, 0.30),
                        (1.5, 0.45),
                    ]
                    
                    for line_width, base_glow_alpha in glow_passes:
                        try:
                            gpu.state.line_width_set(line_width * glow_intensity)
                        except Exception:
                            pass
                        
                        for dash_verts in circle_dash_segments:
                            if len(dash_verts) >= 2:
                                batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": dash_verts})
                                shader.bind()
                                final_alpha = base_glow_alpha * glow_intensity * color_alpha
                                shader.uniform_float("color", (outline_color[0], outline_color[1], outline_color[2], final_alpha))
                                batch.draw(shader)
                    
                    # Draw main dashes with anti-aliasing simulation (multiple passes with slight offsets)
                    # Draw multiple passes with reduced opacity for smoother appearance
                    offsets = [(0, 0), (0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]
                    for offset_x, offset_y in offsets:
                        try:
                            gpu.state.line_width_set(1.5)
                        except Exception:
                            pass
                        for dash_verts in circle_dash_segments:
                            if len(dash_verts) >= 2:
                                # Apply offset to vertices for anti-aliasing
                                offset_verts = [(x + offset_x, y + offset_y) for x, y in dash_verts]
                                batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": offset_verts})
                                shader.bind()
                                alpha = 0.3 if offset_x != 0 or offset_y != 0 else 0.5
                                final_alpha = alpha * color_alpha
                                shader.uniform_float("color", (outline_color[0], outline_color[1], outline_color[2], final_alpha))
                                batch.draw(shader)
            elif use_dashed:
                # Draw dashed pattern
                dash_length = CIRCLE_DASH_LENGTH
                gap_length = CIRCLE_GAP_LENGTH
                dash_list = []
                
                # Calculate circumference and create dash pattern
                circumference = 2 * math.pi * radius
                if circumference > 0:
                    pattern_length = dash_length + gap_length
                    angle_per_pixel = (2 * math.pi) / circumference
                    
                    pos = 0.0
                    while pos < circumference:
                        # Draw dash
                        dash_end = min(pos + dash_length, circumference)
                        start_angle = pos * angle_per_pixel
                        end_angle = dash_end * angle_per_pixel
                        
                        # Create smooth segments for this dash (more segments for smoother appearance)
                        num_dash_segs = max(5, int((dash_end - pos) / (circumference / num_segments)))
                        dash_verts = []
                        for j in range(num_dash_segs + 1):
                            t = j / num_dash_segs if num_dash_segs > 0 else 0
                            angle = start_angle + (end_angle - start_angle) * t
                            x = cx + radius * math.cos(angle)
                            y = cy + radius * math.sin(angle)
                            dash_verts.append((x, y))
                        
                        # Add this dash to the list
                        if len(dash_verts) >= 2:
                            dash_list.append(dash_verts)
                        
                        pos += pattern_length
                
                # Get outline color from preferences
                prefs = _get_addon_preferences()
                outline_color = tuple(prefs.selection_outline_color) if prefs else (1.0, 1.0, 1.0, 1.0)
                # Ensure 4 components (RGBA) - handle old 3-component colors
                if len(outline_color) == 3:
                    outline_color = outline_color + (1.0,)
                
                # Draw each dash separately
                for dash_verts in dash_list:
                    if len(dash_verts) >= 2:
                        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": dash_verts})
                        shader.bind()
                        color_alpha = outline_color[3] if len(outline_color) > 3 else 1.0
                        shader.uniform_float("color", (outline_color[0], outline_color[1], outline_color[2], 1.0 * color_alpha))
                        batch.draw(shader)
            else:
                # Get outline color from preferences
                prefs = _get_addon_preferences()
                outline_color = tuple(prefs.selection_outline_color) if prefs else (1.0, 1.0, 1.0, 1.0)
                # Ensure 4 components (RGBA) - handle old 3-component colors
                if len(outline_color) == 3:
                    outline_color = outline_color + (1.0,)
                
                # Draw solid line with anti-aliasing simulation (multiple passes with slight offsets)
                # Draw multiple passes with reduced opacity for smoother appearance
                color_alpha = outline_color[3] if len(outline_color) > 3 else 1.0
                offsets = [(0, 0), (0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]
                for offset_x, offset_y in offsets:
                    offset_verts = [(x + offset_x, y + offset_y) for x, y in circle_verts]
                    batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": offset_verts})
                    shader.bind()
                    alpha = 0.3 if offset_x != 0 or offset_y != 0 else 0.5
                    final_alpha = alpha * color_alpha
                    shader.uniform_float("color", (outline_color[0], outline_color[1], outline_color[2], final_alpha))
                    batch.draw(shader)
            
            # Draw gradient visualization only in smart mode (solid line, not dashed)
            # Gradient should only show when using smart selection, not standard selection
            # Don't show gradient or percentage when Ctrl mode is active (select all mode doesn't use tolerance)
            if not use_dashed:
                try:
                    tolerance_multiplier = getattr(self, 'tolerance_multiplier', 100.0)
                except (ReferenceError, AttributeError):
                    tolerance_multiplier = 100.0
                ctrl_pressed = getattr(self, '_ctrl_pressed', False)
                prefs = _get_addon_preferences()
                show_gradient = prefs.show_gradient if prefs else True
                
                if tolerance_multiplier != 100.0 and show_gradient and not ctrl_pressed:
                    _draw_gradient_ring(cx, cy, radius, tolerance_multiplier, num_segments)
            
            if not use_dashed:
                # Draw tolerance multiplier text in center of circle when not 100% (only in smart mode - solid line)
                try:
                    tolerance_multiplier = getattr(self, 'tolerance_multiplier', 100.0)
                except (ReferenceError, AttributeError):
                    tolerance_multiplier = 100.0
                ctrl_pressed = getattr(self, '_ctrl_pressed', False)
                prefs = _get_addon_preferences()
                show_percentage = prefs.show_percentage_text if prefs else True
                
                if tolerance_multiplier != 100.0 and show_percentage and not ctrl_pressed:
                    # Get text settings from preferences
                    base_font_size = prefs.text_font_size if prefs else 15
                    text_size_scale = prefs.text_size_scale if prefs else 0.35
                    text_color = prefs.text_color if prefs else (1.0, 1.0, 1.0, 1.0)
                    # Ensure 4 components (RGBA) - handle old 3-component colors
                    if len(text_color) == 3:
                        text_color = text_color + (1.0,)
                    text_color_r, text_color_g, text_color_b = text_color[0], text_color[1], text_color[2]
                    text_placement = prefs.text_placement_circle if prefs else 'MOUSE'
                    
                    # Calculate circle size for dynamic text scaling
                    # Use radius (diameter would be 2*radius, but radius is more intuitive)
                    circle_size = radius * 2.0  # Use diameter for scaling
                    
                    # Calculate dynamic font size: blend between fixed size and scaled size
                    # Scale factor: convert circle size to reasonable font size (e.g., 100px circle = ~15px font)
                    scale_factor = 0.15  # Adjust this to control scaling sensitivity
                    scaled_size = circle_size * scale_factor
                    text_font_size = base_font_size * (1.0 - text_size_scale) + scaled_size * text_size_scale
                    text_font_size = max(10, min(50, text_font_size))  # Clamp between 10 and 50
                    
                    # Calculate opacity based on circle size (fade out when too small)
                    # Minimum size where text is fully visible: around 40 pixels diameter (for readable text)
                    # Below this, fade from 0 (at 0px) to 1.0 (at 40px)
                    min_size_threshold = 40.0
                    if circle_size <= 0:
                        text_alpha = 0.0
                    elif circle_size >= min_size_threshold:
                        text_alpha = 1.0
                    else:
                        # Smooth fade: 0 to 1.0 as size goes from 0 to min_size_threshold
                        # Use a power curve for faster fade (make it disappear quickly when small)
                        normalized_size = circle_size / min_size_threshold
                        text_alpha = pow(normalized_size, 2.5)  # Power curve for faster fade
                    
                    # Don't draw if too transparent
                    if text_alpha < 0.01:
                        pass  # Skip drawing
                    else:
                        # Determine text position based on placement setting
                        # For CENTER: use circle center (cx, cy = self.center)
                        # For MOUSE: use current mouse position if available, otherwise circle center
                        if text_placement == 'CENTER':
                            text_center_x = cx  # Circle center
                            text_center_y = cy
                        else:  # MOUSE
                            # Try to use current mouse position, fallback to circle center
                            current_mouse_val = _safe_getattr(self, 'current_mouse', None)
                            if current_mouse_val is not None:
                                text_center_x, text_center_y = current_mouse_val
                            else:
                                text_center_x = cx
                                text_center_y = cy
                        
                        # Get display mode and format text accordingly
                        # If multiple object types are selected, always show percentage
                        force_percentage = _has_multiple_object_types(self)
                        text_display_mode = prefs.text_display_mode if prefs else 'PERCENTAGE'
                        if text_display_mode == 'VERTEX_COUNT' and not force_percentage:
                            try:
                                threshold_value, obj_type = _get_vertex_count_threshold(self)
                                if obj_type == 'LIGHT':
                                    text = _format_power_display(threshold_value) if threshold_value > 0 else "0W"
                                else:
                                    text = _format_vertex_count_display(threshold_value) if threshold_value > 0 else "0"
                            except (ReferenceError, AttributeError):
                                text = "0"
                        else:
                            text = _format_multiplier_display(tolerance_multiplier)
                        
                        font_id = 0
                        blf.size(font_id, int(text_font_size))
                        
                        # Get text dimensions first
                        try:
                            text_width, text_height = blf.dimensions(font_id, text)
                        except (AttributeError, TypeError):
                            # Fallback: approximate dimensions based on font size and character count
                            text_width = text_font_size * 0.6 * len(text)
                            text_height = text_font_size * 1.2
                        
                        # Calculate text position (center text if CENTER mode)
                        if text_placement == 'CENTER':
                            # Center text: blf.position sets bottom-left, so we need to offset by half text dimensions
                            text_x = text_center_x - text_width / 2.0
                            text_y = text_center_y - text_height / 2.0
                        else:  # MOUSE
                            # Position at mouse cursor (already set above)
                            text_x = text_center_x
                            text_y = text_center_y
                        
                        blf.position(font_id, text_x, text_y, 0)
                        # Multiply calculated alpha by color alpha (if available)
                        color_alpha = text_color[3] if len(text_color) > 3 else 1.0
                        final_text_alpha = text_alpha * color_alpha
                        blf.color(font_id, text_color_r, text_color_g, text_color_b, final_text_alpha)
                        blf.draw(font_id, text)

                try:
                    gpu.state.blend_set('NONE')
                except Exception:
                    pass
        except (ReferenceError, AttributeError):
            # Operator instance has been removed, silently return
            return

    def invoke(self, context, event):
        global _last_circle_custom_modal, _last_circle_toggle_modal

        # Only work in OBJECT mode - otherwise pass through to Blender's default circle select
        if context.mode != 'OBJECT':
            override = _find_3dview_override()
            if override is not None:
                try:
                    bpy.ops.view3d.select_circle(override, 'INVOKE_DEFAULT')
                except Exception:
                    try:
                        bpy.ops.view3d.select_circle('INVOKE_DEFAULT')
                    except Exception:
                        pass
            else:
                try:
                    bpy.ops.view3d.select_circle('INVOKE_DEFAULT')
                except Exception:
                    pass
            return {'CANCELLED'}

        if context.area.type != 'VIEW_3D':
            return {'CANCELLED'}

        # Gather reference objects: either all selected (in view layer) or the active object as fallback
        refs = [o for o in context.selected_objects if o.name in context.view_layer.objects]
        if not refs and context.active_object and context.active_object.name in context.view_layer.objects:
            refs = [context.active_object]
        if not refs:
            return {'CANCELLED'}

        # If there is a waiting modal, clear it (we are switching into custom)
        if _last_circle_toggle_modal is not None:
            try:
                _last_circle_toggle_modal._should_cancel = True
                if hasattr(_last_circle_toggle_modal, "_handler") and _last_circle_toggle_modal._handler is not None:
                    try:
                        bpy.types.SpaceView3D.draw_handler_remove(_last_circle_toggle_modal._handler, 'WINDOW')
                    except Exception:
                        pass
                    _last_circle_toggle_modal._handler = None
            except Exception:
                pass
            _last_circle_toggle_modal = None

        # armed but not drawing until first click
        global _circle_last_radius, _circle_use_dashed
        # Get defaults from preferences
        prefs = _get_addon_preferences()
        default_radius = prefs.circle_default_radius if prefs else _circle_last_radius
        
        # Use preference default or remembered value (prefer remembered if valid)
        if _circle_last_radius and _circle_last_radius > 0:
            self.radius = _circle_last_radius  # Use remembered radius from last use
        else:
            self.radius = default_radius  # Use preference default
            _circle_last_radius = default_radius  # Initialize global with default
        
        # Always start in dashed mode (standard behavior)
        _circle_use_dashed = True
        self.use_dashed = _circle_use_dashed
        self.center = None
        self.current_mouse = (event.mouse_region_x, event.mouse_region_y)
        self.drawing = False
        self._deselect_mode = False
        self._handler = None
        self.animated_zebra_mode = False  # Flag for animated zebra mode
        self._last_shift_press_time = 0.0  # Track last shift press for double-shift detection
        self._animation_timer = None  # Timer handle for continuous redraw
        self._statusbar_handler = None  # Handler for status bar hints
        self.depsgraph = context.evaluated_depsgraph_get()
        self.use_dashed = True  # Always start in dashed mode
        global _last_tolerance_multiplier_step
        self.tolerance_multiplier_step = _last_tolerance_multiplier_step  # Use remembered value
        self.tolerance_multiplier = _step_index_to_multiplier(self.tolerance_multiplier_step)
        global _global_ctrl_pressed
        self._ctrl_pressed = _global_ctrl_pressed  # Use global state to persist between invocations
        
        # Store initial selection state - objects selected at start should not be deselected
        self._initial_selected = set()
        view_layer_objects = context.view_layer.objects
        for obj in view_layer_objects:
            if obj.select_get():
                self._initial_selected.add(obj.name)

        # compute reference metrics now and store for later matching
        self._ref_metrics = []
        self._ref_objects = []  # Store reference objects directly for light type checking
        self._ref_light_types = {}  # Store light types for reference objects: {ref_obj_name: light_type}
        self._ref_metric_to_obj = {}  # Map (type, metric) to list of reference objects for light type checking
        for o in refs:
            t, m = get_object_metric(o, self.depsgraph, self.use_evaluated)
            self._ref_metrics.append((t, m))
            self._ref_objects.append(o)  # Store reference object directly
            # Store mapping from metric to objects
            metric_key = (t, round(m, 6))
            if metric_key not in self._ref_metric_to_obj:
                self._ref_metric_to_obj[metric_key] = []
            self._ref_metric_to_obj[metric_key].append(o)
            # Store light type if it's a light
            if t == 'LIGHT':
                light_type = get_light_type(o)
                if light_type:
                    self._ref_light_types[o.name] = light_type
            # Store text content if it's a text object
            if t == 'FONT':
                text_content = get_text_content(o)
                if text_content is not None:
                    if not hasattr(self, '_ref_text_contents'):
                        self._ref_text_contents = {}
                    self._ref_text_contents[o.name] = text_content
        
        # Store active object's metric separately for deselection (deselect only based on active object)
        self._active_ref_metrics = []
        self._active_ref_objects = []  # Store active reference object directly for light type and text content checking
        self._active_ref_light_types = {}  # Store light types for active object
        self._active_ref_text_contents = {}  # Store text contents for active object
        self._active_metric_to_obj = {}  # Map (type, metric) to list of active reference objects for light type checking
        if context.active_object and context.active_object.name in context.view_layer.objects:
            t, m = get_object_metric(context.active_object, self.depsgraph, self.use_evaluated)
            self._active_ref_metrics.append((t, m))
            self._active_ref_objects.append(context.active_object)  # Store active object directly
            # Store mapping from metric to objects
            metric_key = (t, round(m, 6))
            if metric_key not in self._active_metric_to_obj:
                self._active_metric_to_obj[metric_key] = []
            self._active_metric_to_obj[metric_key].append(context.active_object)
            # Store light type if it's a light
            if t == 'LIGHT':
                light_type = get_light_type(context.active_object)
                if light_type:
                    self._active_ref_light_types[context.active_object.name] = light_type
            # Store text content if it's a text object
            if t == 'FONT':
                text_content = get_text_content(context.active_object)
                if text_content is not None:
                    self._active_ref_text_contents[context.active_object.name] = text_content

        # Add draw handler immediately to show crosshair (solid)
        self._handler = bpy.types.SpaceView3D.draw_handler_add(self.draw_callback, (context,), 'WINDOW', 'POST_PIXEL')
        
        # Add status bar draw handler to show hints with icons
        # Create a closure that captures the operator instance
        operator_instance = self
        def draw_statusbar_circle(self, context):
            """Draw status bar hints with icons for circle select"""
            # Check if this operator is still active
            global _last_circle_custom_modal
            if _last_circle_custom_modal != operator_instance:
                return  # Only draw if this is the active operator
            
            # Only show tolerance hints when in smart mode (solid line, not dashed)
            use_dashed = getattr(operator_instance, 'use_dashed', True)
            if not use_dashed:
                layout = self.layout
                row = layout.row(align=True)
                row.scale_x = 0.0  # Compact spacing
                
                # Shift+Scroll: Adjust tolerance (smart mode)
                row.label(text="", icon='EVENT_SHIFT')
                row.label(text="+")
                row.label(text="", icon='MOUSE_MMB')
                row.label(text=": Adjust tolerance (smart mode)")
                
                # Alt: Reset to 100%
                row.label(text="", icon='EVENT_ALT')
                row.label(text=": Reset to 100%")
        
        try:
            # Use prepend to add to left side instead of append (which adds to right side)
            self._statusbar_handler = bpy.types.STATUSBAR_HT_header.prepend(draw_statusbar_circle)
        except Exception as e:
            print(f"Error adding status bar handler: {e}")
            pass
        
        context.area.tag_redraw()

        # remember the custom modal instance so pressing C will toggle back
        _last_circle_custom_modal = self

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        global _last_circle_custom_modal, _circle_last_radius, _last_tolerance_multiplier_step, _global_ctrl_pressed

        # If user presses C while this custom modal is active -> toggle between dashed and solid line
        if event.type == 'C' and event.value == 'PRESS':
            global _circle_use_dashed
            # Toggle line style
            _circle_use_dashed = not _circle_use_dashed
            self.use_dashed = _circle_use_dashed
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Handle Shift-Shift to toggle animated zebra mode
        if event.type in {'LEFT_SHIFT', 'RIGHT_SHIFT'} and event.value == 'PRESS':
            current_time = time.time()
            # Check if this is a double-shift (within 0.5 seconds)
            if current_time - self._last_shift_press_time < 0.5:
                # Toggle animated zebra mode
                self.animated_zebra_mode = not self.animated_zebra_mode
                
                # Manage animation timer
                if self.animated_zebra_mode:
                    # Start animation timer - capture context in closure
                    operator_self = self
                    context_ref = context
                    def animation_timer():
                        try:
                            if hasattr(operator_self, 'animated_zebra_mode') and operator_self.animated_zebra_mode:
                                # Force redraw of all 3D viewports
                                for area in context_ref.screen.areas:
                                    if area.type == 'VIEW_3D':
                                        area.tag_redraw()
                                return 0.016  # ~60fps for smooth animation
                            return None  # Stop timer
                        except:
                            return None  # Stop timer on error
                    # Only register if not already registered
                    if self._animation_timer is None:
                        self._animation_timer = bpy.app.timers.register(animation_timer, first_interval=0.016)
                else:
                    # Stop animation timer (it will stop itself when animated_zebra_mode is False)
                    self._animation_timer = None
                
                context.area.tag_redraw()
            self._last_shift_press_time = current_time
            return {'RUNNING_MODAL'}
        
        # Force continuous redraw when animated zebra mode is active
        # (Timer handles continuous redraw, but we also redraw on events for responsiveness)
        if getattr(self, 'animated_zebra_mode', False):
            context.area.tag_redraw()

        # Middle mouse press starts a deselect circle
        if event.type == 'MIDDLEMOUSE' and event.value == 'PRESS' and not self.drawing:
            self.center = (event.mouse_region_x, event.mouse_region_y)
            # Use current radius (which may have been changed with mouse wheel)
            self.current_mouse = self.center
            self.drawing = True
            self._deselect_mode = True
            self._prev_circle_bounds = None  # Track previous frame's circle bounds for deselection
            # Track objects that were deselected and their previous selection state (for restoration)
            self._deselected_objects = {}  # {obj_name: was_selected_before}
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # If MMB released while in deselect mode -> stop drawing but keep circle active
        if event.type == 'MIDDLEMOUSE' and event.value == 'RELEASE' and self.drawing and self._deselect_mode:
            # Deselection was already done continuously during MOUSEMOVE
            # Don't finish - just stop drawing so user can continue deselecting
            # Remember radius (use preference default if not set)
            prefs = _get_addon_preferences()
            default_radius = prefs.circle_default_radius if prefs else 25.0
            _circle_last_radius = getattr(self, 'radius', default_radius)  # Remember radius
            self.drawing = False
            self._deselect_mode = False
            self._prev_circle_bounds = None  # Reset previous circle bounds
            self._deselected_objects = {}  # Clear deselected objects tracking
            self.center = None  # Clear center to prevent ghost circle
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # cancel on Esc / RightMouse press
        if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            # Remember the radius before canceling
            _circle_last_radius = getattr(self, 'radius', 25.0)
            if self._handler is not None:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(self._handler, 'WINDOW')
                except Exception:
                    pass
                self._handler = None
            # Remove status bar handler
            if self._statusbar_handler is not None:
                try:
                    bpy.types.STATUSBAR_HT_header.remove(self._statusbar_handler)
                except:
                    pass
                self._statusbar_handler = None
            # Stop animation timer if active
            self.animated_zebra_mode = False  # This will cause the timer to stop itself
            _last_circle_custom_modal = None
            context.area.tag_redraw()
            return {'CANCELLED'}

        # start on left press (arm drawing)
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and not self.drawing:
            self.center = (event.mouse_region_x, event.mouse_region_y)
            # Use current radius (which may have been changed with mouse wheel)
            # Don't reset to 25.0, keep the current size
            self.current_mouse = self.center
            self.drawing = True
            self._deselect_mode = False
            self._last_selected = set()  # Track what was selected to avoid re-selecting
            self._prev_circle_bounds = None  # Track previous frame's circle bounds for selection
            self._selected_by_circle = set()  # Track objects selected by circle (they stay selected even if circle moves away)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Handle Ctrl key press - toggle mode (press to toggle on/off)
        if event.type in {'LEFT_CTRL', 'RIGHT_CTRL'} and event.value == 'PRESS':
            t_start = time.perf_counter()
            # Toggle Ctrl mode
            _global_ctrl_pressed = not _global_ctrl_pressed
            self._ctrl_pressed = _global_ctrl_pressed
            
            # When Ctrl mode is activated, tolerance adjustment is disabled but the value is remembered
            # (Ctrl mode selects all, so tolerance filtering is not used, but we keep the user's setting)
            
            # Immediately re-evaluate all objects in current circle (if drawing)
            if self.drawing and hasattr(self, 'center') and self.center is not None and hasattr(self, 'radius'):
                region = context.region
                rv3d = context.region_data
                cx, cy = self.center
                radius = self.radius
                
                objects_processed = 0
                objects_in_circle = 0
                t_metric_total = 0.0
                t_matching_total = 0.0
                t_selection_total = 0.0
                matching_calls = 0
                selection_calls = 0
                
                # Re-evaluate all objects in current circle
                # Valid object types
                valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
                # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
                view_layer_objects = context.view_layer.objects
                
                for obj in view_layer_objects:
                    if obj.type not in valid_types:
                        continue
                    if obj.hide_viewport:
                        continue
                    co_world = obj.matrix_world.translation
                    co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                    if co_2d is None:
                        continue
                    x, y = co_2d
                    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                    if dist_sq <= radius ** 2:
                        objects_in_circle += 1
                        objects_processed += 1
                        
                        t_metric_start = time.perf_counter()
                        obj_type, obj_metric = get_object_metric(obj, self.depsgraph, self.use_evaluated)
                        t_metric_total += time.perf_counter() - t_metric_start
                        
                        if self._ctrl_pressed:
                            # Ctrl mode ON: select all same-type objects
                            ref_types = [r_type for r_type, _ in self._ref_metrics]
                            if obj_type in ref_types:
                                t_sel_start = time.perf_counter()
                                try:
                                    obj.select_set(True)
                                    selection_calls += 1
                                except Exception:
                                    pass
                                t_selection_total += time.perf_counter() - t_sel_start
                        else:
                            # Ctrl mode OFF: re-evaluate with normal criteria
                            # For lights, use specialized matching function that checks light type FIRST
                            t_match_start = time.perf_counter()
                            if obj_type == 'LIGHT' and not self._ctrl_pressed:
                                # Use specialized function that checks light type before power
                                matches = _matches_light_with_type_check(obj, obj_type, obj_metric, self._ref_objects, self._ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                matching_calls += 1
                            elif obj_type == 'FONT' and not self._ctrl_pressed:
                                # Use specialized function that checks text content before metric
                                matches = _matches_text_with_content_check(obj, obj_type, obj_metric, self._ref_objects, self._ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                matching_calls += 1
                            else:
                                # For non-lights, non-text, or Ctrl mode, use normal filtering
                                if not self.include_other_types:
                                    # Use pre-computed filtered cache instead of list comprehension
                                    if not hasattr(self, '_ref_metrics_by_type') or self._ref_metrics_by_type is None:
                                        # Fallback: build cache on first use
                                        self._ref_metrics_by_type = {}
                                        for r_type, r_metric in self._ref_metrics:
                                            if r_type not in self._ref_metrics_by_type:
                                                self._ref_metrics_by_type[r_type] = []
                                            self._ref_metrics_by_type[r_type].append((r_type, r_metric))
                                    refs_to_check = self._ref_metrics_by_type.get(obj_type, [])
                                    if not refs_to_check:
                                        t_sel_start = time.perf_counter()
                                        try:
                                            obj.select_set(False)
                                            selection_calls += 1
                                        except Exception:
                                            pass
                                        t_selection_total += time.perf_counter() - t_sel_start
                                        continue
                                    matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                    matching_calls += 1
                                else:
                                    # Use cached list instead of creating new one
                                    if not hasattr(self, '_ref_metrics_list_cache') or self._ref_metrics_list_cache is None:
                                        self._ref_metrics_list_cache = list(self._ref_metrics)
                                    refs_to_check = self._ref_metrics_list_cache
                                    matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                    matching_calls += 1
                            t_matching_total += time.perf_counter() - t_match_start
                            
                            t_sel_start = time.perf_counter()
                            try:
                                obj.select_set(matches)
                                selection_calls += 1
                            except Exception:
                                pass
                            t_selection_total += time.perf_counter() - t_sel_start
                
                t_total = time.perf_counter() - t_start
                print(f"[C Selection - CTRL Toggle] Total: {t_total*1000:.2f}ms | "
                      f"Objects: {objects_processed} processed, {objects_in_circle} in circle | "
                      f"Get metrics: {t_metric_total*1000:.2f}ms | "
                      f"Matching: {t_matching_total*1000:.2f}ms ({matching_calls} calls) | "
                      f"Selection: {t_selection_total*1000:.2f}ms ({selection_calls} calls)")
                
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Handle Alt key to reset tolerance multiplier back to 100%
        if event.type == 'LEFT_ALT' or event.type == 'RIGHT_ALT':
            if event.value == 'PRESS':
                t_start = time.perf_counter()
                use_dashed = getattr(self, 'use_dashed', True)
                # Only reset tolerance when using smart selection (solid line mode)
                if not use_dashed:
                    self.tolerance_multiplier_step = -1  # Reset to 100%
                    self.tolerance_multiplier = 100.0
                    _last_tolerance_multiplier_step = -1  # Remember the reset
                    
                    # Live reselection/redeselection: if we have an active circle, reapply selection/deselection with new tolerance (100%)
                    if hasattr(self, 'center') and self.center is not None and hasattr(self, 'radius'):
                        region = context.region
                        rv3d = context.region_data
                        cx, cy = self.center
                        radius = self.radius
                        
                        objects_processed = 0
                        objects_in_circle = 0
                        t_metric_total = 0.0
                        t_matching_total = 0.0
                        t_selection_total = 0.0
                        matching_calls = 0
                        selection_calls = 0
                        
                        # If in deselect mode, reapply deselection with 100% tolerance
                        if self._deselect_mode:
                            # Only restore objects that are currently INSIDE the circle
                            # Objects outside the circle should remain deselected and not be affected
                            objects_to_restore = {}  # Track which objects to restore (only those in circle)
                            # Valid object types
                            valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
                            # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
                            view_layer_objects = context.view_layer.objects
                            
                            for obj in view_layer_objects:
                                if obj.type not in valid_types:
                                    continue
                                if obj.hide_viewport:
                                    continue
                                co_world = obj.matrix_world.translation
                                co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                                if co_2d is None:
                                    continue
                                x, y = co_2d
                                dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                                # Only restore objects that are currently in the circle
                                if dist_sq <= radius ** 2 and obj.name in self._deselected_objects:
                                    was_selected = self._deselected_objects[obj.name]
                                    objects_to_restore[obj.name] = was_selected
                                    try:
                                        obj.select_set(was_selected)
                                    except Exception:
                                        pass
                            
                            # Remove restored objects from tracking (they'll be re-evaluated)
                            for obj_name in objects_to_restore:
                                del self._deselected_objects[obj_name]
                            
                            # Reapply deselection with 100% tolerance (only to objects currently in circle)
                            # Valid object types
                            valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
                            # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
                            view_layer_objects = context.view_layer.objects
                            
                            for obj in view_layer_objects:
                                if obj.type not in valid_types:
                                    continue
                                if obj.hide_viewport:
                                    continue
                                objects_processed += 1
                                co_world = obj.matrix_world.translation
                                co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                                if co_2d is None:
                                    continue
                                x, y = co_2d
                                dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                                if dist_sq <= radius ** 2:
                                    objects_in_circle += 1
                                    t_metric_start = time.perf_counter()
                                    obj_type, obj_metric = get_object_metric(obj, self.depsgraph, self.use_evaluated)
                                    t_metric_total += time.perf_counter() - t_metric_start
                                    
                                    # Use active object's metrics for deselection
                                    if not self._active_ref_metrics:
                                        continue  # No active object to deselect against
                                    
                                    # For lights, use specialized matching function that checks light type FIRST
                                    t_match_start = time.perf_counter()
                                    if obj_type == 'LIGHT' and not self._ctrl_pressed:
                                        # Use specialized function that checks light type before power
                                        matches = _matches_light_with_type_check(obj, obj_type, obj_metric, self._active_ref_objects, self._active_ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                        matching_calls += 1
                                    elif obj_type == 'FONT' and not self._ctrl_pressed:
                                        # Use specialized function that checks text content before metric
                                        matches = _matches_text_with_content_check(obj, obj_type, obj_metric, self._active_ref_objects, self._active_ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                        matching_calls += 1
                                    else:
                                        # For non-lights, non-text, or Ctrl mode, use normal filtering
                                        if not self.include_other_types:
                                            if not any(r_type == obj_type for r_type, _ in self._active_ref_metrics):
                                                continue
                                            # Use pre-computed filtered cache instead of list comprehension
                                            if not hasattr(self, '_active_ref_metrics_by_type') or self._active_ref_metrics_by_type is None:
                                                # Fallback: build cache on first use
                                                self._active_ref_metrics_by_type = {}
                                                for r_type, r_metric in self._active_ref_metrics:
                                                    if r_type not in self._active_ref_metrics_by_type:
                                                        self._active_ref_metrics_by_type[r_type] = []
                                                    self._active_ref_metrics_by_type[r_type].append((r_type, r_metric))
                                            refs_to_check = self._active_ref_metrics_by_type.get(obj_type, [])
                                        else:
                                            # Use cached list instead of creating new one
                                            if not hasattr(self, '_active_ref_metrics_list_cache') or self._active_ref_metrics_list_cache is None:
                                                self._active_ref_metrics_list_cache = list(self._active_ref_metrics)
                                            refs_to_check = self._active_ref_metrics_list_cache
                                        matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                        matching_calls += 1
                                    t_matching_total += time.perf_counter() - t_match_start
                                    
                                    # If object matches smart selection criteria (based on active object), deselect it
                                    if matches:
                                        # Store previous selection state before deselecting
                                        t_sel_start = time.perf_counter()
                                        if obj.name not in self._deselected_objects:
                                            self._deselected_objects[obj.name] = obj.select_get()
                                        try:
                                            obj.select_set(False)
                                            selection_calls += 1
                                        except Exception:
                                            pass
                                        t_selection_total += time.perf_counter() - t_sel_start
                        else:
                            # Normal selection mode: reselect objects in circle with 100% tolerance
                            # Valid object types
                            valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
                            # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
                            view_layer_objects = context.view_layer.objects
                            
                            for obj in view_layer_objects:
                                if obj.type not in valid_types:
                                    continue
                                if obj.hide_viewport:
                                    continue
                                objects_processed += 1
                                co_world = obj.matrix_world.translation
                                co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                                if co_2d is None:
                                    continue
                                x, y = co_2d
                                dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                                if dist_sq <= radius ** 2:
                                    objects_in_circle += 1
                                    t_metric_start = time.perf_counter()
                                    obj_type, obj_metric = get_object_metric(obj, self.depsgraph, self.use_evaluated)
                                    t_metric_total += time.perf_counter() - t_metric_start
                                    
                                    # Check if Ctrl is held - select all objects of same type regardless of parameters
                                    if self._ctrl_pressed:
                                        # Check if this object type matches any of the reference object types
                                        ref_types = [r_type for r_type, _ in self._ref_metrics]
                                        if obj_type in ref_types:
                                            t_sel_start = time.perf_counter()
                                            try:
                                                obj.select_set(True)
                                                self._selected_by_circle.add(obj.name)  # Track that this object was selected by circle
                                                selection_calls += 1
                                            except Exception:
                                                pass
                                            t_selection_total += time.perf_counter() - t_sel_start
                                        continue
                                    
                                    if not self.include_other_types:
                                        # Use pre-computed filtered cache instead of list comprehension
                                        if self._ref_metrics_by_type is None:
                                            # Fallback: build cache on first use
                                            self._ref_metrics_by_type = {}
                                            for r_type, r_metric in self._ref_metrics:
                                                if r_type not in self._ref_metrics_by_type:
                                                    self._ref_metrics_by_type[r_type] = []
                                                self._ref_metrics_by_type[r_type].append((r_type, r_metric))
                                        refs_to_check = self._ref_metrics_by_type.get(obj_type, [])
                                        if not refs_to_check:
                                            continue
                                    else:
                                        # Use cached list instead of creating new one
                                        if self._ref_metrics_list_cache is None:
                                            self._ref_metrics_list_cache = list(self._ref_metrics)
                                        refs_to_check = self._ref_metrics_list_cache
                                    
                                    # For lights, use specialized matching function that checks light type FIRST
                                    t_match_start = time.perf_counter()
                                    if obj_type == 'LIGHT' and not self._ctrl_pressed:
                                        # Use specialized function that checks light type before power
                                        matches = _matches_light_with_type_check(obj, obj_type, obj_metric, self._ref_objects, self._ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                        matching_calls += 1
                                    else:
                                        # For non-lights or Ctrl mode, use normal matching
                                        matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                        matching_calls += 1
                                    t_matching_total += time.perf_counter() - t_match_start
                                    
                                    if matches:
                                        t_sel_start = time.perf_counter()
                                        try:
                                            obj.select_set(True)
                                            self._selected_by_circle.add(obj.name)  # Track that this object was selected by circle
                                            selection_calls += 1
                                        except Exception:
                                            pass
                                        t_selection_total += time.perf_counter() - t_sel_start
                    
                    t_total = time.perf_counter() - t_start
                    mode_str = "DESELECT" if self._deselect_mode else "SELECT"
                    print(f"[C Selection - ALT Reset ({mode_str})] Total: {t_total*1000:.2f}ms | "
                          f"Objects: {objects_processed} processed, {objects_in_circle} in circle | "
                          f"Get metrics: {t_metric_total*1000:.2f}ms | "
                          f"Matching: {t_matching_total*1000:.2f}ms ({matching_calls} calls) | "
                          f"Selection: {t_selection_total*1000:.2f}ms ({selection_calls} calls)")
                    
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        # Handle Shift + scroll wheel to adjust tolerance multiplier (only when using smart selection)
        if event.shift and (event.type == 'WHEELUPMOUSE' or event.type == 'WHEELDOWNMOUSE') and event.value == 'PRESS':
            use_dashed = getattr(self, 'use_dashed', True)
            # Only adjust tolerance when using smart selection (solid line mode)
            if not use_dashed:
                # If Ctrl mode is active, turn it off (tolerance adjustment is incompatible with "select all" mode)
                if self._ctrl_pressed:
                    _global_ctrl_pressed = False
                    self._ctrl_pressed = False
                
                direction = 1 if event.type == 'WHEELUPMOUSE' else -1
                old_multiplier = getattr(self, 'tolerance_multiplier', 100.0)
                self.tolerance_multiplier_step, self.tolerance_multiplier = _adjust_tolerance_multiplier(self.tolerance_multiplier_step, direction)
                # Remember the new tolerance step for next time
                _last_tolerance_multiplier_step = self.tolerance_multiplier_step
                
                # Live reselection/redeselection: if we have an active circle, reapply selection/deselection with new tolerance
                if hasattr(self, 'center') and self.center is not None and hasattr(self, 'radius'):
                    region = context.region
                    rv3d = context.region_data
                    cx, cy = self.center
                    radius = self.radius
                    
                    # If in deselect mode, reapply deselection with new tolerance
                    if self._deselect_mode:
                        # Only restore objects that are currently INSIDE the circle
                        # Objects outside the circle should remain deselected and not be affected
                        objects_to_restore = {}  # Track which objects to restore (only those in circle)
                        # Valid object types
                        valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
                        # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
                        view_layer_objects = context.view_layer.objects
                        
                        for obj in view_layer_objects:
                            if obj.type not in valid_types:
                                continue
                            if obj.hide_viewport:
                                continue
                            co_world = obj.matrix_world.translation
                            co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                            if co_2d is None:
                                continue
                            x, y = co_2d
                            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                            # Only restore objects that are currently in the circle
                            if dist_sq <= radius ** 2 and obj.name in self._deselected_objects:
                                was_selected = self._deselected_objects[obj.name]
                                objects_to_restore[obj.name] = was_selected
                                try:
                                    obj.select_set(was_selected)
                                except Exception:
                                    pass
                        
                        # Remove restored objects from tracking (they'll be re-evaluated)
                        for obj_name in objects_to_restore:
                            del self._deselected_objects[obj_name]
                        
                        # Reapply deselection with new tolerance (only to objects currently in circle)
                        # Valid object types
                        valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
                        # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
                        view_layer_objects = context.view_layer.objects
                        
                        for obj in view_layer_objects:
                            if obj.type not in valid_types:
                                continue
                            if obj.hide_viewport:
                                continue
                            co_world = obj.matrix_world.translation
                            co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                            if co_2d is None:
                                continue
                            x, y = co_2d
                            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                            if dist_sq <= radius ** 2:
                                obj_type, obj_metric = get_object_metric(obj, self.depsgraph, self.use_evaluated)
                                
                                # Use active object's metrics for deselection
                                if not self._active_ref_metrics:
                                    continue  # No active object to deselect against
                                
                                # For lights, use specialized matching function that checks light type FIRST
                                if obj_type == 'LIGHT' and not self._ctrl_pressed:
                                    # Use specialized function that checks light type before power
                                    matches = _matches_light_with_type_check(obj, obj_type, obj_metric, self._active_ref_objects, self._active_ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                elif obj_type == 'FONT' and not self._ctrl_pressed:
                                    # Use specialized function that checks text content before metric
                                    matches = _matches_text_with_content_check(obj, obj_type, obj_metric, self._active_ref_objects, self._active_ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                else:
                                    # For non-lights, non-text, or Ctrl mode, use normal filtering
                                    if not self.include_other_types:
                                        if not any(r_type == obj_type for r_type, _ in self._active_ref_metrics):
                                            continue
                                        # Use pre-computed filtered cache instead of list comprehension
                                    if getattr(self, '_active_ref_metrics_by_type', None) is None:
                                        # Fallback: build cache on first use
                                        self._active_ref_metrics_by_type = {}
                                        for r_type, r_metric in self._active_ref_metrics:
                                            if r_type not in self._active_ref_metrics_by_type:
                                                self._active_ref_metrics_by_type[r_type] = []
                                            self._active_ref_metrics_by_type[r_type].append((r_type, r_metric))
                                        refs_to_check = self._active_ref_metrics_by_type.get(obj_type, [])
                                        matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                    else:
                                        # Use cached list instead of creating new one
                                        if getattr(self, '_active_ref_metrics_list_cache', None) is None:
                                            self._active_ref_metrics_list_cache = list(self._active_ref_metrics)
                                        refs_to_check = self._active_ref_metrics_list_cache
                                        matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                
                                # If object matches smart selection criteria (based on active object), deselect it
                                if matches:
                                    # Store previous selection state before deselecting
                                    if obj.name not in self._deselected_objects:
                                        self._deselected_objects[obj.name] = obj.select_get()
                                    try:
                                        obj.select_set(False)
                                    except Exception:
                                        pass
                    else:
                        # Normal selection mode: reselect objects in circle with new tolerance
                        # Valid object types
                        valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
                        # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
                        view_layer_objects = context.view_layer.objects
                        
                        for obj in view_layer_objects:
                            if obj.type not in valid_types:
                                continue
                            if obj.hide_viewport:
                                continue
                            co_world = obj.matrix_world.translation
                            co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                            if co_2d is None:
                                continue
                            x, y = co_2d
                            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                            if dist_sq <= radius ** 2:
                                obj_type, obj_metric = get_object_metric(obj, self.depsgraph, self.use_evaluated)
                                
                                # For lights, use specialized matching function that checks light type FIRST
                                if obj_type == 'LIGHT' and not self._ctrl_pressed:
                                    # Use specialized function that checks light type before power
                                    matches = _matches_light_with_type_check(obj, obj_type, obj_metric, self._ref_objects, self._ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                else:
                                    # For non-lights or Ctrl mode, use normal filtering
                                    if not self.include_other_types:
                                        # Use pre-computed filtered cache instead of list comprehension
                                        if self._ref_metrics_by_type is None:
                                            # Fallback: build cache on first use
                                            self._ref_metrics_by_type = {}
                                        for r_type, r_metric in self._ref_metrics:
                                            if r_type not in self._ref_metrics_by_type:
                                                self._ref_metrics_by_type[r_type] = []
                                            self._ref_metrics_by_type[r_type].append((r_type, r_metric))
                                        refs_to_check = self._ref_metrics_by_type.get(obj_type, [])
                                        if not refs_to_check:
                                            continue
                                        matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                    else:
                                        # Use cached list instead of creating new one
                                        if self._ref_metrics_list_cache is None:
                                            self._ref_metrics_list_cache = list(self._ref_metrics)
                                        refs_to_check = self._ref_metrics_list_cache
                                        matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                if matches:
                                    try:
                                        obj.select_set(True)
                                        self._selected_by_circle.add(obj.name)  # Track that this object was selected by circle
                                    except Exception:
                                        pass
                
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Handle mouse wheel to scale circle up/down (step size increases with circle size)
        if event.type == 'WHEELUPMOUSE' and event.value == 'PRESS' and not event.shift:
            # Increase circle radius - step size scales with current radius
            # For small circles: small steps, for large circles: large steps
            step = max(5.0, self.radius * 0.1)  # 10% of current radius, minimum 5px
            self.radius = min(self.radius + step, 500.0)  # Max 500 pixels
            _circle_last_radius = self.radius  # Remember the size
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        
        if event.type == 'WHEELDOWNMOUSE' and event.value == 'PRESS' and not event.shift:
            # Decrease circle radius - step size scales with current radius
            step = max(5.0, self.radius * 0.1)  # 10% of current radius, minimum 5px
            self.radius = max(self.radius - step, 5.0)  # Min 5 pixels
            _circle_last_radius = self.radius  # Remember the size
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        
        # update while moving - circle follows mouse, selects/deselects objects continuously
        if event.type == 'MOUSEMOVE':
            t_start = time.perf_counter()
            self.current_mouse = (event.mouse_region_x, event.mouse_region_y)
            if self.drawing and not self._deselect_mode:
                # Circle center follows mouse, radius stays constant
                mx, my = self.current_mouse
                self.center = (mx, my)  # Circle follows mouse
                # Radius stays constant (set on mouse press)
                
                # Continuously select objects as circle moves (always continuous for circle selection)
                region = context.region
                rv3d = context.region_data
                cx, cy = self.center
                radius = self.radius
                use_dashed = getattr(self, 'use_dashed', True)
                
                # Track objects currently in circle
                current_circle_objects = set()
                
                # Performance tracking
                t_iter_start = time.perf_counter()
                objects_processed = 0
                objects_in_circle = 0
                t_coord_conv_total = 0.0
                t_metric_total = 0.0
                t_matching_total = 0.0
                t_selection_total = 0.0
                t_filter_total = 0.0
                matching_calls = 0
                selection_calls = 0
                
                # Valid object types
                valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
                
                # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
                t_filter_start = time.perf_counter()
                view_layer_objects = context.view_layer.objects
                t_filter_total += time.perf_counter() - t_filter_start
                
                # Only process objects currently in the circle - don't deselect objects that leave
                for obj in view_layer_objects:
                    t_filter_start = time.perf_counter()
                    if obj.type not in valid_types:
                        t_filter_total += time.perf_counter() - t_filter_start
                        continue
                    if obj.hide_viewport:
                        t_filter_total += time.perf_counter() - t_filter_start
                        continue
                    t_filter_total += time.perf_counter() - t_filter_start
                    objects_processed += 1
                    
                    t_coord_start = time.perf_counter()
                    co_world = obj.matrix_world.translation
                    co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                    t_coord_conv_total += time.perf_counter() - t_coord_start
                    
                    if co_2d is None:
                        continue
                    x, y = co_2d
                    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                    if dist_sq <= radius ** 2:
                        objects_in_circle += 1
                        current_circle_objects.add(obj.name)
                        
                        t_metric_start = time.perf_counter()
                        obj_type, obj_metric = get_object_metric(obj, self.depsgraph, self.use_evaluated)
                        t_metric_total += time.perf_counter() - t_metric_start
                        
                        # Dashed mode: select all objects in circle
                        # Solid mode: use similarity matching
                        if use_dashed:
                            # Select all objects within circle (no similarity check)
                            t_sel_start = time.perf_counter()
                            try:
                                obj.select_set(True)
                                self._selected_by_circle.add(obj.name)  # Track that this object was selected by circle
                                selection_calls += 1
                            except Exception:
                                pass
                            t_selection_total += time.perf_counter() - t_sel_start
                        else:
                            # Use similarity matching (original behavior)
                            # Check if Ctrl is held - select all objects of same type regardless of parameters
                            if self._ctrl_pressed:
                                # Check if this object type matches any of the reference object types
                                # Use cached ref types set instead of list comprehension
                                if not hasattr(self, '_ref_types_set') or self._ref_types_set is None:
                                    self._ref_types_set = {r_type for r_type, _ in self._ref_metrics}
                                if obj_type in self._ref_types_set:
                                    t_sel_start = time.perf_counter()
                                    try:
                                        obj.select_set(True)
                                        self._selected_by_circle.add(obj.name)  # Track that this object was selected by circle
                                        selection_calls += 1
                                    except Exception:
                                        pass
                                    t_selection_total += time.perf_counter() - t_sel_start
                            else:
                                # Normal mode: use similarity matching
                                if not self.include_other_types:
                                    # Use pre-computed filtered cache instead of list comprehension
                                    if not hasattr(self, '_ref_metrics_by_type') or self._ref_metrics_by_type is None:
                                        # Fallback: build cache on first use
                                        self._ref_metrics_by_type = {}
                                        for r_type, r_metric in self._ref_metrics:
                                            if r_type not in self._ref_metrics_by_type:
                                                self._ref_metrics_by_type[r_type] = []
                                            self._ref_metrics_by_type[r_type].append((r_type, r_metric))
                                    refs_to_check = self._ref_metrics_by_type.get(obj_type, [])
                                    if not refs_to_check:
                                        continue
                                else:
                                    # Use cached list instead of creating new one
                                    if not hasattr(self, '_ref_metrics_list_cache') or self._ref_metrics_list_cache is None:
                                        self._ref_metrics_list_cache = list(self._ref_metrics)
                                    refs_to_check = self._ref_metrics_list_cache
                                
                                # For lights, use specialized matching function that checks light type FIRST
                                t_match_start = time.perf_counter()
                                if obj_type == 'LIGHT' and not self._ctrl_pressed:
                                    matches = _matches_light_with_type_check(obj, obj_type, obj_metric, self._ref_objects, self._ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                # For text objects, use specialized matching function that checks text content FIRST
                                elif obj_type == 'FONT' and not self._ctrl_pressed:
                                    matches = _matches_text_with_content_check(obj, obj_type, obj_metric, self._ref_objects, self._ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                else:
                                    # For other objects, use normal matching
                                    matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                t_matching_total += time.perf_counter() - t_match_start
                                matching_calls += 1
                                
                                if matches:
                                    t_sel_start = time.perf_counter()
                                    try:
                                        obj.select_set(True)
                                        self._selected_by_circle.add(obj.name)  # Track that this object was selected by circle
                                        selection_calls += 1
                                    except Exception:
                                        pass
                                    t_selection_total += time.perf_counter() - t_sel_start
                
                t_iter_total = time.perf_counter() - t_iter_start
                t_total = time.perf_counter() - t_start
                t_unaccounted = t_total - t_coord_conv_total - t_metric_total - t_matching_total - t_selection_total - t_filter_total
                
                # Report performance metrics
                print(f"[C Selection - SELECT] Total: {t_total*1000:.2f}ms | "
                      f"Objects: {objects_processed} processed, {objects_in_circle} in circle | "
                      f"Filtering: {t_filter_total*1000:.2f}ms | "
                      f"Coord conv: {t_coord_conv_total*1000:.2f}ms | "
                      f"Get metrics: {t_metric_total*1000:.2f}ms | "
                      f"Matching: {t_matching_total*1000:.2f}ms ({matching_calls} calls) | "
                      f"Selection: {t_selection_total*1000:.2f}ms ({selection_calls} calls) | "
                      f"Unaccounted: {t_unaccounted*1000:.2f}ms")
                
                # Store current circle bounds for next frame
                self._prev_circle_bounds = (cx, cy, radius)
            elif self.drawing and self._deselect_mode:
                # Circle center follows mouse, continuously deselect objects
                mx, my = self.current_mouse
                self.center = (mx, my)  # Circle follows mouse
                
                region = context.region
                rv3d = context.region_data
                cx, cy = self.center
                radius = self.radius
                use_dashed = getattr(self, 'use_dashed', True)
                
                # Don't restore objects that were deselected - they should stay deselected
                # Once an object is deselected by the circle, it remains deselected even when circle moves away
                
                # Track objects currently in circle
                current_circle_objects = set()
                
                # Performance tracking
                t_iter_start = time.perf_counter()
                objects_processed = 0
                objects_in_circle = 0
                t_coord_conv_total = 0.0
                t_metric_total = 0.0
                t_matching_total = 0.0
                t_selection_total = 0.0
                t_filter_total = 0.0
                matching_calls = 0
                selection_calls = 0
                
                # Valid object types
                valid_types = {'MESH', 'EMPTY', 'CURVE', 'SURFACE', 'FONT', 'META', 'ARMATURE', 'LIGHT', 'CAMERA'}
                
                # Use view_layer.objects instead of scene.objects - already filters hidden/out-of-layer objects
                t_filter_start = time.perf_counter()
                view_layer_objects = context.view_layer.objects
                t_filter_total += time.perf_counter() - t_filter_start
                
                # Perform smart deselection on objects currently in circle
                # Use only active object as reference for deselection (not all selected objects)
                for obj in view_layer_objects:
                    t_filter_start = time.perf_counter()
                    if obj.type not in valid_types:
                        t_filter_total += time.perf_counter() - t_filter_start
                        continue
                    if obj.hide_viewport:
                        t_filter_total += time.perf_counter() - t_filter_start
                        continue
                    t_filter_total += time.perf_counter() - t_filter_start
                    objects_processed += 1
                    
                    t_coord_start = time.perf_counter()
                    co_world = obj.matrix_world.translation
                    co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, co_world)
                    t_coord_conv_total += time.perf_counter() - t_coord_start
                    
                    if co_2d is None:
                        continue
                    x, y = co_2d
                    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                    if dist_sq <= radius ** 2:
                        objects_in_circle += 1
                        current_circle_objects.add(obj.name)
                        # Dashed mode: deselect all objects in circle
                        # Solid mode: use similarity matching for deselection
                        # When user explicitly uses middle mouse, they can deselect anything
                        if use_dashed:
                            # Deselect all objects within circle (no similarity check)
                            # Store previous selection state before deselecting
                            t_sel_start = time.perf_counter()
                            if obj.name not in self._deselected_objects:
                                self._deselected_objects[obj.name] = obj.select_get()
                            try:
                                obj.select_set(False)
                                selection_calls += 1
                            except Exception:
                                pass
                            t_selection_total += time.perf_counter() - t_sel_start
                        else:
                            # Use similarity matching for deselection (based on active object only)
                            t_metric_start = time.perf_counter()
                            obj_type, obj_metric = get_object_metric(obj, self.depsgraph, self.use_evaluated)
                            t_metric_total += time.perf_counter() - t_metric_start
                            
                            # Use active object's metrics for deselection (not all selected objects)
                            if not self._active_ref_metrics:
                                continue  # No active object to deselect against
                            
                            # For lights, use specialized matching function that checks light type FIRST
                            t_match_start = time.perf_counter()
                            if obj_type == 'LIGHT' and not self._ctrl_pressed:
                                # Use specialized function that checks light type before power
                                matches = _matches_light_with_type_check(obj, obj_type, obj_metric, self._active_ref_objects, self._active_ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                matching_calls += 1
                            elif obj_type == 'FONT' and not self._ctrl_pressed:
                                # Use specialized function that checks text content before metric
                                matches = _matches_text_with_content_check(obj, obj_type, obj_metric, self._active_ref_objects, self._active_ref_metrics, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier, self.depsgraph, self.use_evaluated)
                                matching_calls += 1
                            else:
                                # For non-lights, non-text, or Ctrl mode, use normal filtering
                                if not self.include_other_types:
                                    if not any(r_type == obj_type for r_type, _ in self._active_ref_metrics):
                                        matches = False
                                    else:
                                        # Use pre-computed filtered cache instead of list comprehension
                                        if getattr(self, '_active_ref_metrics_by_type', None) is None:
                                            # Fallback: build cache on first use
                                            self._active_ref_metrics_by_type = {}
                                            for r_type, r_metric in self._active_ref_metrics:
                                                if r_type not in self._active_ref_metrics_by_type:
                                                    self._active_ref_metrics_by_type[r_type] = []
                                                self._active_ref_metrics_by_type[r_type].append((r_type, r_metric))
                                        refs_to_check = self._active_ref_metrics_by_type.get(obj_type, [])
                                        matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                        matching_calls += 1
                                else:
                                    # Use cached list instead of creating new one
                                    if getattr(self, '_active_ref_metrics_list_cache', None) is None:
                                        self._active_ref_metrics_list_cache = list(self._active_ref_metrics)
                                    refs_to_check = self._active_ref_metrics_list_cache
                                    matches = _matches_any_reference(obj_type, obj_metric, refs_to_check, self.use_percentage, self.percent_tolerance, self.tolerance, self.tolerance_multiplier)
                                    matching_calls += 1
                            t_matching_total += time.perf_counter() - t_match_start
                            
                            # If object matches smart selection criteria (based on active object), deselect it
                            # When user explicitly uses middle mouse, they can deselect anything
                            if matches:
                                # Store previous selection state before deselecting
                                t_sel_start = time.perf_counter()
                                if obj.name not in self._deselected_objects:
                                    self._deselected_objects[obj.name] = obj.select_get()
                                try:
                                    obj.select_set(False)
                                    selection_calls += 1
                                except Exception:
                                    pass
                                t_selection_total += time.perf_counter() - t_sel_start
                
                t_iter_total = time.perf_counter() - t_iter_start
                t_total = time.perf_counter() - t_start
                t_unaccounted = t_total - t_coord_conv_total - t_metric_total - t_matching_total - t_selection_total - t_filter_total
                
                # Report performance metrics
                print(f"[C Selection - DESELECT] Total: {t_total*1000:.2f}ms | "
                      f"Objects: {objects_processed} processed, {objects_in_circle} in circle | "
                      f"Filtering: {t_filter_total*1000:.2f}ms | "
                      f"Coord conv: {t_coord_conv_total*1000:.2f}ms | "
                      f"Get metrics: {t_metric_total*1000:.2f}ms | "
                      f"Matching: {t_matching_total*1000:.2f}ms ({matching_calls} calls) | "
                      f"Selection: {t_selection_total*1000:.2f}ms ({selection_calls} calls) | "
                      f"Unaccounted: {t_unaccounted*1000:.2f}ms")
                
                # Store current circle bounds for next frame
                self._prev_circle_bounds = (cx, cy, radius)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # finish on left release - but keep circle active for continued selection
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self.drawing and not self._deselect_mode:
            # Selection was already done continuously during MOUSEMOVE (circle selection is always continuous)
            # Don't finish - just stop drawing so user can continue selecting
            # Remember radius (use preference default if not set)
            prefs = _get_addon_preferences()
            default_radius = prefs.circle_default_radius if prefs else 25.0
            _circle_last_radius = getattr(self, 'radius', default_radius)  # Remember radius
            self.drawing = False
            self._prev_circle_bounds = None  # Reset previous circle bounds
            self._selected_by_circle = set()  # Clear selected objects tracking (they stay selected, but we stop tracking)
            self.center = None  # Clear center to prevent ghost circle
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}


class VIEW3D_OT_circle_select_toggle(bpy.types.Operator):
    """Press C to activate smart circle select - circle follows mouse and selects similar objects."""
    bl_idname = "view3d.circle_select_toggle"
    bl_label = "Circle Select Toggle"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        global _last_circle_toggle_modal, _last_circle_custom_modal

        # Only work in OBJECT mode
        if context.mode != 'OBJECT':
            # Pass through to standard circle select in non-object mode
            override = _find_3dview_override()
            if override is not None:
                try:
                    bpy.ops.view3d.select_circle(override, 'INVOKE_DEFAULT')
                except Exception:
                    try:
                        bpy.ops.view3d.select_circle('INVOKE_DEFAULT')
                    except Exception:
                        pass
            else:
                try:
                    bpy.ops.view3d.select_circle('INVOKE_DEFAULT')
                except Exception:
                    pass
            return {'CANCELLED'}

        # Single C press: activate smart circle select
        try:
            return bpy.ops.view3d.select_circle_similar_verts('INVOKE_DEFAULT')
        except Exception:
            return {'CANCELLED'}


class ALEX_SMART_SELECTION_AddonPreferences(bpy.types.AddonPreferences):
    """Addon preferences for Alex Smart Selection"""
    # bl_idname MUST match the MODULE NAME (filename without .py), NOT bl_info["name"]
    # For single-file addons, Blender uses the filename as the module identifier
    # If installed as Alex_Smart_Selection_10.py, bl_idname should be "Alex_Smart_Selection_10"
    # If installed as Alex_Smart_Selection_5.py, bl_idname should be "Alex_Smart_Selection_5"
    bl_idname = __name__.split('.')[-1]  # Use the actual module name from __name__
    
    # Gradient visualization settings
    gradient_steps: IntProperty(
        name="Gradient Steps",
        description="Number of ring layers for smooth gradient (higher = smoother but more CPU intensive)",
        default=50,  # Match GRADIENT_STEPS = 50
        min=5,
        max=50,
    )
    
    gradient_max_alpha: FloatProperty(
        name="Maximum Gradient Alpha",
        description="Maximum transparency/opacity of the gradient (0.0 = transparent, 1.0 = opaque)",
        default=0.25,
        min=0.01,
        max=1.0,
        step=0.01,
        precision=2,
    )
    
    gradient_curve_power: FloatProperty(
        name="Gradient Curve Power",
        description="Controls how fast gradient increases with deviation (lower = faster growth/more visible at 70%-800%, higher = slower growth). Use lower values for better visibility at small deviations",
        default=0.7,
        min=0.1,
        max=2.0,
        step=0.05,
        precision=3,
    )
    
    gradient_min_extent_inner: FloatProperty(
        name="Min Extent Inner (%)",
        description="Minimum gradient extent inward (as percentage of radius) for values < 100%. Increase for better visibility at values like 70%",
        default=3.0,
        min=0.1,
        max=50.0,
        step=0.5,
        precision=1,
    )
    
    gradient_max_extent_inner: FloatProperty(
        name="Max Extent Inner (%)",
        description="Maximum gradient extent inward (as percentage of radius) for values < 100%",
        default=15.0,
        min=1.0,
        max=90.0,
        step=1.0,
        precision=1,
    )
    
    gradient_min_extent_outer: FloatProperty(
        name="Min Extent Outer (%)",
        description="Minimum gradient extent outward (as percentage of radius) for values > 100%. Increase for better visibility at values like 800%",
        default=3.0,
        min=0.1,
        max=50.0,
        step=0.5,
        precision=1,
    )
    
    gradient_max_extent_outer: FloatProperty(
        name="Max Extent Outer (%)",
        description="Maximum gradient extent outward (as percentage of radius) for values > 100%",
        default=15.0,
        min=1.0,
        max=100.0,
        step=1.0,
        precision=1,
    )
    
    gradient_min_intensity: FloatProperty(
        name="Minimum Intensity",
        description="Minimum gradient intensity/alpha (0.0 = transparent, 1.0 = opaque). Higher values make gradient more visible at small deviations like 70% or 800%",
        default=0.15,
        min=0.0,
        max=0.5,
        step=0.01,
        precision=2,
    )
    
    gradient_max_cap_percentage: FloatProperty(
        name="Max Cap Percentage",
        description="Maximum percentage value for calculating gradient (e.g., 15000 means gradient scales from 100% to 15000%+)",
        default=15000.0,
        min=150.0,
        max=1000000.0,
        step=1000.0,
        precision=1,
    )
    
    gradient_ease: FloatProperty(
        name="Gradient Ease",
        description="Controls how smooth/easy the gradient fade is (higher = smoother/easier fade, lower = sharper fade)",
        default=2.5,
        min=0.5,
        max=5.0,
        step=0.1,
        precision=1,
    )
    
    # Dash pattern settings
    zebra_dash_pixels: IntProperty(
        name="Zebra Crosshair Dash (pixels)",
        description="Length of each dash segment in the zebra crosshair",
        default=3,
        min=1,
        max=20,
    )
    
    zebra_gap_pixels: IntProperty(
        name="Zebra Crosshair Gap (pixels)",
        description="Length of gap between dashes in the zebra crosshair",
        default=0,
        min=0,
        max=20,
    )
    
    box_dash_length: FloatProperty(
        name="Box Dash Length (pixels)",
        description="Length of each dash segment for box selection outline",
        default=3.0,
        min=1.0,
        max=20.0,
        step=0.5,
        precision=1,
    )
    
    box_gap_length: FloatProperty(
        name="Box Gap Length (pixels)",
        description="Length of gap between dashes for box selection outline",
        default=3.0,
        min=0.0,
        max=20.0,
        step=0.5,
        precision=1,
    )
    
    animated_zebra_speed: FloatProperty(
        name="Animated Zebra Speed (pixels/second)",
        description="Animation speed for animated zebra mode (how fast dashes move around the selection box)",
        default=18.0,
        min=0.1,
        max=1000.0,
        step=1.0,
        precision=1,
    )
    
    # Animated Zebra Mode specific settings (separate from regular box selection)
    animated_zebra_dash_length: FloatProperty(
        name="Animated Zebra Dash Length (pixels)",
        description="Length of each dash segment for animated zebra mode",
        default=10.0,
        min=0.5,
        max=100.0,
        step=0.5,
        precision=1,
    )
    
    animated_zebra_gap_length: FloatProperty(
        name="Animated Zebra Gap Length (pixels)",
        description="Length of gap between dashes for animated zebra mode",
        default=10.0,
        min=0.5,
        max=100.0,
        step=0.5,
        precision=1,
    )
    
    
    circle_dash_length: FloatProperty(
        name="Circle Dash Length (pixels)",
        description="Length of each dash segment for circle selection outline",
        default=3.0,
        min=1.0,
        max=20.0,
        step=0.5,
        precision=1,
    )
    
    circle_gap_length: FloatProperty(
        name="Circle Gap Length (pixels)",
        description="Length of gap between dashes for circle selection outline",
        default=3.0,
        min=0.0,
        max=20.0,
        step=0.5,
        precision=1,
    )
    
    # Circle selection settings
    circle_double_press_threshold: FloatProperty(
        name="Circle Double Press Threshold (seconds)",
        description="Time window for detecting double press of C key (lower = faster, higher = more forgiving)",
        default=0.25,
        min=0.1,
        max=1.0,
        step=0.05,
        precision=2,
    )
    
    circle_default_radius: FloatProperty(
        name="Default Circle Radius (pixels)",
        description="Default radius for circle selection when first activated",
        default=25.0,
        min=5.0,
        max=500.0,
        step=1.0,
        precision=1,
    )
    
    # Display toggles
    show_gradient: BoolProperty(
        name="Show Gradient",
        description="Enable gradient visualization when adjusting tolerance multiplier",
        default=True,
    )
    
    show_percentage_text: BoolProperty(
        name="Show Percentage Text",
        description="Display tolerance percentage number in center of selection",
        default=True,
    )
    
    text_display_mode: EnumProperty(
        name="Text Display Mode",
        description="What to display in the text: percentage or vertex count",
        items=[
            ('PERCENTAGE', 'Percentage', 'Display tolerance percentage (e.g., 150%)', 0),
            ('VERTEX_COUNT', 'Vertex Count', 'Display selected vertex count (e.g., 1250)', 1),
        ],
        default='VERTEX_COUNT',
    )
    
    # Text settings
    text_font_size: IntProperty(
        name="Text Font Size",
        description="Font size for percentage text display",
        default=15,
        min=10,
        max=50,
    )
    
    text_color: FloatVectorProperty(
        name="Text Color",
        description="Text color (RGBA)",
        default=(1.0, 1.0, 1.0, 1.0),
        min=0.0,
        max=1.0,
        subtype='COLOR',
        size=4,
    )
    
    text_placement_circle: EnumProperty(
        name="Circle Text Placement",
        description="Where to position the percentage text for circle selection",
        items=[
            ('CENTER', 'Center', 'Center text in circle center', 0),
            ('MOUSE', 'Mouse Position', 'Position text at mouse cursor', 1),
        ],
        default='MOUSE',
    )
    
    # Performance optimization settings (Box Selection only - Circle selection is always continuous)
    auto_disable_continuous_selection: BoolProperty(
        name="Auto-Disable Continuous Selection (Box Only)",
        description="Automatically disable continuous selection updates for BOX selection when too many objects are detected. Selection will only apply when mouse button is released (like standard B selection). Circle selection is always continuous and not affected by this setting.",
        default=True,
    )
    
    max_objects_for_continuous: IntProperty(
        name="Max Objects for Continuous Selection",
        description="Maximum number of objects in scene before disabling continuous selection updates for BOX selection. If scene has more objects, selection only applies on mouse release",
        default=1000,
        min=100,
        max=10000,
        step=100,
    )
    
    max_time_for_continuous: FloatProperty(
        name="Max Time for Continuous Selection (ms)",
        description="Maximum calculation time (milliseconds) before disabling continuous selection for BOX selection. If selection takes longer, it will only apply on mouse release",
        default=50.0,
        min=10.0,
        max=500.0,
        step=5.0,
        precision=1,
    )
    
    text_placement_box: EnumProperty(
        name="Box Text Placement",
        description="Where to position the percentage text for box selection",
        items=[
            ('CENTER', 'Center', 'Center text in box center', 0),
            ('MOUSE', 'Mouse Position', 'Position text at mouse cursor', 1),
        ],
        default='CENTER',
    )
    
    text_size_scale: FloatProperty(
        name="Text Size Scale",
        description="Scale factor for text size relative to selection size (0.0 = fixed size, 1.0 = fully scales with selection)",
        default=0.35,
        min=0.0,
        max=1.0,
        step=0.05,
        precision=2,
    )
    
    gradient_color: FloatVectorProperty(
        name="Gradient Color",
        description="Color for gradient visualization (RGBA)",
        default=(1.0, 1.0, 1.0, 1.0),
        min=0.0,
        max=1.0,
        subtype='COLOR',
        size=4,
    )
    
    selection_outline_color: FloatVectorProperty(
        name="Selection Outline Color",
        description="Color for circle and box selection outlines (RGBA)",
        default=(1.0, 1.0, 1.0, 1.0),
        min=0.0,
        max=1.0,
        subtype='COLOR',
        size=4,
    )
    
    def draw(self, context):
        layout = self.layout
        
        # Circle Selection - Compact
        box_circle = layout.box()
        row = box_circle.row()
        row.prop(self, "circle_default_radius", text="Default Radius")
        layout.separator()
        
        # Display Options - Buttons
        box_display = layout.box()
        box_display.label(text="Display Options:", icon='PREFERENCES')
        row = box_display.row()
        # Use toggle buttons (depress shows pressed state)
        op1 = row.operator("alex_smart_selection.toggle_gradient", text="Gradient ON" if self.show_gradient else "Gradient OFF", depress=self.show_gradient, icon='SHADING_SOLID' if self.show_gradient else 'SHADING_WIRE')
        op2 = row.operator("alex_smart_selection.toggle_text", text="% Text ON" if self.show_percentage_text else "% Text OFF", depress=self.show_percentage_text, icon='FONT_DATA' if self.show_percentage_text else 'FORCE_CURVE')
        layout.separator()
        
        # Gradient Settings - Only shown if enabled
        if self.show_gradient:
            box_gradient = layout.box()
            row = box_gradient.row()
            row.label(text="Gradient Settings:", icon='SHADING_SOLID')
            split = box_gradient.split(factor=0.5)
            col = split.column()
            col.prop(self, "gradient_steps", text="Steps")
            col.prop(self, "gradient_max_alpha", text="Max Alpha")
            col.prop(self, "gradient_curve_power", text="Curve Power")
            col = split.column()
            col.prop(self, "gradient_min_intensity", text="Min Intensity")
            col.prop(self, "gradient_max_cap_percentage", text="Max Cap %")
            col.prop(self, "gradient_ease", text="Ease")
            
            # Extent Settings - Compact (side by side)
            row = box_gradient.row()
            split = row.split(factor=0.5)
            col_inner = split.column()
            col_inner.label(text="Inner:")
            col_inner.prop(self, "gradient_min_extent_inner", text="Min %")
            col_inner.prop(self, "gradient_max_extent_inner", text="Max %")
            split = split.split()
            col_outer = split.column()
            col_outer.label(text="Outer:")
            col_outer.prop(self, "gradient_min_extent_outer", text="Min %")
            col_outer.prop(self, "gradient_max_extent_outer", text="Max %")
            
            # Gradient color
            row = box_gradient.row()
            split = row.split(factor=0.15)
            split.label(text="Color:")
            split.prop(self, "gradient_color", text="")
            layout.separator()
        
        # Text Settings - Only shown if enabled
        if self.show_percentage_text:
            box_text = layout.box()
            box_text.label(text="Text Settings:", icon='FONT_DATA')
            
            # Display mode - own row with split to prevent stretching
            row = box_text.row()
            split = row.split(factor=0.35)
            split.label(text="Display:")
            split.prop(self, "text_display_mode", text="")
            
            # Font size and scale - separate row
            row = box_text.row()
            row.prop(self, "text_font_size", text="Font Size")
            row.prop(self, "text_size_scale", text="Size Scale")
            
            # Color - own row with proper spacing
            row = box_text.row()
            split = row.split(factor=0.15)
            split.label(text="Color:")
            split.prop(self, "text_color", text="")
            
            # Placement - separate rows with split to prevent stretching
            row = box_text.row()
            split = row.split(factor=0.35)
            split.label(text="Box Placement:")
            split.prop(self, "text_placement_box", text="")
            row = box_text.row()
            split = row.split(factor=0.35)
            split.label(text="Circle Placement:")
            split.prop(self, "text_placement_circle", text="")
            layout.separator()
        
        # Dash Patterns - Compact (all in one box)
        box_dash = layout.box()
        row = box_dash.row(align=True)
        row.label(text="Dash Patterns:", icon='LINENUMBERS_ON')
        split = box_dash.split(factor=0.33)
        col1 = split.column()
        col1.label(text="Zebra:")
        col1.prop(self, "zebra_dash_pixels", text="Dash")
        col1.prop(self, "zebra_gap_pixels", text="Gap")
        split = split.split(factor=0.5)
        col2 = split.column()
        col2.label(text="Box:")
        col2.prop(self, "box_dash_length", text="Dash")
        col2.prop(self, "box_gap_length", text="Gap")
        split = split.split()
        col3 = split.column()
        col3.label(text="Circle:")
        col3.prop(self, "circle_dash_length", text="Dash")
        col3.prop(self, "circle_gap_length", text="Gap")
        
        # Animated Zebra Mode Settings - Separate section
        layout.separator()
        box_animated = layout.box()
        box_animated.label(text="Animated Zebra Mode Settings:", icon='ANIM')
        row1 = box_animated.row()
        row1.prop(self, "animated_zebra_speed", text="Speed")
        row1.prop(self, "animated_zebra_dash_length", text="Dash Length")
        row2 = box_animated.row()
        row2.prop(self, "animated_zebra_gap_length", text="Gap Length")
        
        # Selection outline color
        row = box_dash.row()
        split = row.split(factor=0.15)
        split.label(text="Outline Color:")
        split.prop(self, "selection_outline_color", text="")
        layout.separator()
        
        # Performance Optimization Settings (Box Selection only)
        box_perf = layout.box()
        box_perf.label(text="Performance Optimization (Box Selection Only):", icon='PREFERENCES')
        box_perf.label(text="Note: Circle selection is always continuous and not affected by these settings", icon='INFO')
        box_perf.prop(self, "auto_disable_continuous_selection", text="Auto-Disable Continuous Selection")
        if self.auto_disable_continuous_selection:
            col = box_perf.column()
            col.prop(self, "max_objects_for_continuous", text="Max Objects")
            col.prop(self, "max_time_for_continuous", text="Max Time (ms)")
        layout.separator()


class ALEX_SMART_SELECTION_OT_toggle_gradient(bpy.types.Operator):
    """Toggle gradient display on/off"""
    bl_idname = "alex_smart_selection.toggle_gradient"
    bl_label = "Toggle Gradient"
    bl_options = {'INTERNAL'}
    
    def execute(self, context):
        try:
            addon_name = __name__.split('.')[-1]
            if addon_name in context.preferences.addons:
                prefs = context.preferences.addons[addon_name].preferences
                prefs.show_gradient = not prefs.show_gradient
            else:
                # Try bl_info name
                addon_name = bl_info.get("name", "")
                if addon_name and addon_name in context.preferences.addons:
                    prefs = context.preferences.addons[addon_name].preferences
                    prefs.show_gradient = not prefs.show_gradient
        except (KeyError, AttributeError):
            pass
        return {'FINISHED'}


class ALEX_SMART_SELECTION_OT_toggle_text(bpy.types.Operator):
    """Toggle percentage text display on/off"""
    bl_idname = "alex_smart_selection.toggle_text"
    bl_label = "Toggle Text"
    bl_options = {'INTERNAL'}
    
    def execute(self, context):
        try:
            addon_name = __name__.split('.')[-1]
            if addon_name in context.preferences.addons:
                prefs = context.preferences.addons[addon_name].preferences
                prefs.show_percentage_text = not prefs.show_percentage_text
            else:
                # Try bl_info name
                addon_name = bl_info.get("name", "")
                if addon_name and addon_name in context.preferences.addons:
                    prefs = context.preferences.addons[addon_name].preferences
                    prefs.show_percentage_text = not prefs.show_percentage_text
        except (KeyError, AttributeError):
            pass
        return {'FINISHED'}


classes = (
    ALEX_SMART_SELECTION_AddonPreferences,
    ALEX_SMART_SELECTION_OT_toggle_gradient,
    ALEX_SMART_SELECTION_OT_toggle_text,
    VIEW3D_OT_box_select_similar_verts,
    VIEW3D_OT_box_select_toggle,
    VIEW3D_OT_circle_select_similar_verts,
    VIEW3D_OT_circle_select_toggle,
)


addon_keymaps = []


def register():
    # Register preferences class first (important for single-file addons)
    try:
        bpy.utils.register_class(ALEX_SMART_SELECTION_AddonPreferences)
    except ValueError:
        try:
            bpy.utils.unregister_class(ALEX_SMART_SELECTION_AddonPreferences)
        except:
            pass
        bpy.utils.register_class(ALEX_SMART_SELECTION_AddonPreferences)
    except Exception as e:
        print(f"Error registering preferences: {e}")
    
    # Register other classes
    for cls in classes:
        if cls == ALEX_SMART_SELECTION_AddonPreferences:
            continue  # Already registered
        try:
            bpy.utils.register_class(cls)
        except ValueError as e:
            # Class might already be registered, try unregistering first
            try:
                bpy.utils.unregister_class(cls)
            except:
                pass
            bpy.utils.register_class(cls)

    wm = bpy.context.window_manager
    if wm.keyconfigs.addon:
        km = wm.keyconfigs.addon.keymaps.new(name='3D View', space_type='VIEW_3D')
        kmi_box = km.keymap_items.new('view3d.box_select_toggle', 'B', 'PRESS')
        addon_keymaps.append((km, kmi_box))
        kmi_circle = km.keymap_items.new('view3d.circle_select_toggle', 'C', 'PRESS')
        addon_keymaps.append((km, kmi_circle))


def unregister():
    global _last_toggle_modal, _last_custom_modal, _last_circle_toggle_modal, _last_circle_custom_modal
    
    # Clear status bar text if any operators are active
    try:
        bpy.context.workspace.status_text_set(None)
    except:
        pass
    
    # try to cancel any lingering waiting modal (box)
    if _last_toggle_modal is not None:
        try:
            _last_toggle_modal._should_cancel = True
            if hasattr(_last_toggle_modal, "_handler") and _last_toggle_modal._handler is not None:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(_last_toggle_modal._handler, 'WINDOW')
                except Exception:
                    pass
                _last_toggle_modal._handler = None
        except Exception:
            pass
        _last_toggle_modal = None

    # try to cancel any lingering custom modal (box)
    if _last_custom_modal is not None:
        try:
            _last_custom_modal._should_cancel = True
            if hasattr(_last_custom_modal, "_handler") and _last_custom_modal._handler is not None:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(_last_custom_modal._handler, 'WINDOW')
                except Exception:
                    pass
                _last_custom_modal._handler = None
            if hasattr(_last_custom_modal, "_statusbar_handler") and _last_custom_modal._statusbar_handler is not None:
                try:
                    bpy.types.STATUSBAR_HT_header.remove(_last_custom_modal._statusbar_handler)
                except:
                    pass
        except Exception:
            pass
        _last_custom_modal = None

    # try to cancel any lingering waiting modal (circle)
    if _last_circle_toggle_modal is not None:
        try:
            _last_circle_toggle_modal._should_cancel = True
            if hasattr(_last_circle_toggle_modal, "_handler") and _last_circle_toggle_modal._handler is not None:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(_last_circle_toggle_modal._handler, 'WINDOW')
                except Exception:
                    pass
                _last_circle_toggle_modal._handler = None
        except Exception:
            pass
        _last_circle_toggle_modal = None

    # try to cancel any lingering custom modal (circle)
    if _last_circle_custom_modal is not None:
        try:
            _last_circle_custom_modal._should_cancel = True
            if hasattr(_last_circle_custom_modal, "_handler") and _last_circle_custom_modal._handler is not None:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(_last_circle_custom_modal._handler, 'WINDOW')
                except Exception:
                    pass
                _last_circle_custom_modal._handler = None
            if hasattr(_last_circle_custom_modal, "_statusbar_handler") and _last_circle_custom_modal._statusbar_handler is not None:
                try:
                    bpy.types.STATUSBAR_HT_header.remove(_last_circle_custom_modal._statusbar_handler)
                except:
                    pass
        except Exception:
            pass
        _last_circle_custom_modal = None

    for cls in classes:
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

    for km, kmi in addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    addon_keymaps.clear()


if __name__ == "__main__":
    register()
