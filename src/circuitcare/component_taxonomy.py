"""Shared component labels for PCB detection, review, and diagnosis."""

from __future__ import annotations

COMPONENT_CLASSES = (
    "battery", "switch", "buzzer", "capacitor", "clock", "connector", "diode",
    "display", "fuse", "heatsink", "ic", "inductor", "led", "potentiometer",
    "relay", "resistor", "transformer", "transistor",
)
UNKNOWN_CLASS = "unknown"
NOT_COMPONENT_CLASS = "not_component"
COMPONENT_LABELS = (*COMPONENT_CLASSES, UNKNOWN_CLASS, NOT_COMPONENT_CLASS)

# Qwen-only recognition layer. These are deliberately excluded from
# COMPONENT_CLASSES so detector label IDs and the YOLO class order stay stable.
QWEN_EXTENSION_CLASSES = ("other_component", "board_feature")
QWEN_REVIEW_CLASSES = (*COMPONENT_CLASSES, *QWEN_EXTENSION_CLASSES,
                       UNKNOWN_CLASS, NOT_COMPONENT_CLASS)

QWEN_SUBTYPES: dict[str, tuple[str, ...]] = {
    "ic": ("mcu", "memory", "regulator", "driver", "operational_amplifier",
           "communication_ic", "other_ic"),
    "connector": ("fpc_ffc", "usb", "hdmi", "pin_header", "terminal_block",
                  "board_to_board", "dc_power_jack", "other_connector"),
    "transistor": ("mosfet", "bjt"),
    "capacitor": ("mlcc", "electrolytic", "tantalum"),
    "inductor": ("chip_inductor", "power_inductor", "ferrite_bead"),
    "switch": ("tactile", "slide", "dip"),
    "clock": ("crystal", "oscillator"),
    "other_component": ("sensor", "optocoupler", "antenna", "module",
                         "ic_socket", "battery_holder", "other_physical_component"),
    "board_feature": ("test_point", "pad", "solder_joint", "via", "trace_damage_suspect"),
}

QWEN_EXTENSION_DEFINITIONS = {
    "other_component": "A physical electronic part outside the existing 18 YOLO classes.",
    "board_feature": "A non-component PCB structure relevant to inspection, not a diagnosis component.",
    "ic_socket": "An empty socket or IC carrier; not an installed integrated circuit.",
}

# Human-readable definitions for every subtype; the mapping key's parent is the
# corresponding key in QWEN_SUBTYPES.
QWEN_SUBTYPE_DEFINITIONS = {
    "mcu": "Programmable microcontroller IC with integrated processing and peripherals.",
    "memory": "Non-volatile or volatile memory IC used to store data or firmware.",
    "regulator": "Power-management IC that regulates or converts supply voltage.",
    "driver": "IC that drives a motor, load, display, or other output stage.",
    "operational_amplifier": "Analog amplifier IC with op-amp package/function evidence.",
    "communication_ic": "IC primarily used for wired or wireless signal communication.",
    "other_ic": "Integrated circuit whose visible function subtype is not distinguishable.",
    "fpc_ffc": "Flat-flex cable connector with a visible flexible-cable insertion slot.",
    "usb": "Connector with a USB-standard receptacle or plug geometry.",
    "hdmi": "Connector with HDMI-standard receptacle geometry.",
    "pin_header": "Connector made of one or more rows of exposed header pins.",
    "terminal_block": "Wire connector with screw, clamp, or spring terminal openings.",
    "board_to_board": "Mating connector designed to connect two circuit boards.",
    "dc_power_jack": "Barrel-style DC power input jack.",
    "other_connector": "Physical connector whose specific family is not visually identifiable.",
    "mosfet": "Field-effect transistor package with MOSFET-specific visual or marking evidence.",
    "bjt": "Bipolar junction transistor package with BJT-specific visual or marking evidence.",
    "mlcc": "Small multilayer ceramic capacitor, commonly an unmarked ceramic chip package.",
    "electrolytic": "Polarized can-style or leaded electrolytic capacitor package.",
    "tantalum": "Polarized tantalum capacitor package identified by body/marking evidence.",
    "chip_inductor": "Small surface-mount inductor package used in signal or power paths.",
    "power_inductor": "Larger magnetic energy-storage inductor package, often shielded.",
    "ferrite_bead": "Ferrite suppression bead used for high-frequency noise filtering.",
    "tactile": "Momentary tactile push-button switch package.",
    "slide": "Switch operated by a visible sliding actuator.",
    "dip": "Multi-position dual-inline switch package with individual toggles.",
    "crystal": "Passive quartz resonator without an integrated oscillator circuit.",
    "oscillator": "Active clock source package containing an oscillator circuit.",
    "sensor": "Physical transducer that senses environmental or board conditions.",
    "optocoupler": "Optically isolated signal-transfer package.",
    "antenna": "RF antenna element or antenna connector identified by visible structure.",
    "module": "Multi-part electronic assembly packaged as a functional module.",
    "ic_socket": "Socket or carrier for an IC; distinguish an empty socket from an installed IC.",
    "battery_holder": "Mechanical holder or clip that retains a replaceable battery.",
    "other_physical_component": "Physical electronic part outside the other listed subtypes.",
    "test_point": "Dedicated exposed test point for probing during inspection or production.",
    "pad": "Exposed conductive land not clearly identified as a separate component.",
    "solder_joint": "Visible solder connection between a lead/pad or two conductive surfaces.",
    "via": "Plated through-layer interconnect hole or via feature.",
    "trace_damage_suspect": "Visible PCB trace region with a possible break or physical damage.",
}

CLASS_ALIASES = {
    "button": "switch", "integrated_circuit": "ic", "unknown_component": "unknown",
    "IC": "ic", "LED": "led",
}


def normalize_component_class(value: object, *, allow_not_component: bool = True) -> str:
    """Return a canonical shared label, retaining ambiguity as ``unknown``."""
    raw = str(value or "").strip()
    canonical = CLASS_ALIASES.get(raw, CLASS_ALIASES.get(raw.casefold(), raw.casefold()))
    allowed = COMPONENT_LABELS if allow_not_component else (*COMPONENT_CLASSES, UNKNOWN_CLASS)
    return canonical if canonical in allowed else UNKNOWN_CLASS


def normalize_qwen_class(value: object) -> str:
    """Normalize a Qwen proposal without widening the detector taxonomy."""
    raw = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {"other": "other_component", "pcb_feature": "board_feature",
               "ic_socket": "other_component", "integrated_circuit": "ic"}
    raw = aliases.get(raw, raw)
    return raw if raw in QWEN_REVIEW_CLASSES else UNKNOWN_CLASS


def normalize_qwen_subtype(base_class: object, subtype: object) -> str | None:
    """Keep only known class-specific subtype suggestions; never infer one."""
    base = normalize_qwen_class(base_class)
    raw = str(subtype or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if not raw or raw in {"none", "null", "unknown", "unspecified"}:
        return None
    # A socket is a distinct physical part, while retaining the parent class.
    if raw == "ic_socket" and base == "other_component":
        return raw
    return raw if raw in QWEN_SUBTYPES.get(base, ()) else None


def review_taxonomy(row: dict) -> dict[str, str | None]:
    """Resolve detector, Qwen, user-confirmed, and safe effective labels once.

    ``effective_class`` is suitable for display/Registry storage. A confirmed
    board feature or extension that the diagnosis engine cannot reason about
    maps to ``unknown`` for diagnosis while its original taxonomy remains in
    the returned fields.
    """
    detector = normalize_component_class(
        row.get("detector_class") or row.get("소자 종류") or "unknown", allow_not_component=False
    )
    qwen_base = normalize_qwen_class(
        row.get("qwen_base_class") or row.get("qwen_component_type_candidate") or "unknown"
    )
    qwen_subtype = normalize_qwen_subtype(
        qwen_base, row.get("qwen_subtype") or row.get("qwen_component_subtype_candidate")
    )
    confirmed_base = row.get("confirmed_base_class")
    confirmed_subtype = row.get("confirmed_subtype")
    if confirmed_base:
        confirmed_base = normalize_qwen_class(confirmed_base)
        confirmed_subtype = normalize_qwen_subtype(confirmed_base, confirmed_subtype)
    elif row.get("qwen_component_type_status") == "USER_CONFIRMED":
        confirmed_base, confirmed_subtype = qwen_base, qwen_subtype
    elif row.get("qwen_component_type_status") == "USER_CONFIRMED_EXTENSION":
        # Legacy rows can lack the explicit confirmed fields. Preserve their
        # extension meaning and never reinterpret the old detector class.
        confirmed_base, confirmed_subtype = qwen_base, qwen_subtype

    effective = confirmed_base or detector
    if effective in {"other_component", "board_feature", "not_component"}:
        diagnosis_class = "unknown"
    else:
        diagnosis_class = normalize_component_class(effective, allow_not_component=False)
    return {
        "detector_class": detector,
        "qwen_base_class": qwen_base if qwen_base != "unknown" else None,
        "qwen_subtype": qwen_subtype,
        "confirmed_base_class": confirmed_base,
        "confirmed_subtype": confirmed_subtype,
        "effective_class": effective,
        "diagnosis_class": diagnosis_class,
        "component_type_status": (
            "USER_CONFIRMED_EXTENSION" if confirmed_base in QWEN_EXTENSION_CLASSES
            else "USER_CONFIRMED" if confirmed_base else "AI_PROPOSED_UNVERIFIED"
            if qwen_base != "unknown" else "UNCONFIRMED"
        ),
        "presence_status": str(row.get("presence_status") or "UNCONFIRMED"),
    }
