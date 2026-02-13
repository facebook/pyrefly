from typing import Any

__all__ = [
    "pole_zero_numerical_data",
    "pole_zero_plot",
    "step_response_numerical_data",
    "step_response_plot",
    "impulse_response_numerical_data",
    "impulse_response_plot",
    "ramp_response_numerical_data",
    "ramp_response_plot",
    "bode_magnitude_numerical_data",
    "bode_phase_numerical_data",
    "bode_magnitude_plot",
    "bode_phase_plot",
    "bode_plot",
]
matplotlib = ...
numpy = ...

def pole_zero_numerical_data(system) -> tuple[Any, Any]: ...
def pole_zero_plot(
    system, pole_color=..., pole_markersize=..., zero_color=..., zero_markersize=..., grid=..., show_axes=..., show=..., **kwargs
) -> None: ...
def step_response_numerical_data(
    system, prec=..., lower_limit=..., upper_limit=..., **kwargs
) -> tuple[Any, Any] | tuple[list[Any], list[Any]]: ...
def step_response_plot(
    system, color=..., prec=..., lower_limit=..., upper_limit=..., show_axes=..., grid=..., show=..., **kwargs
) -> None: ...
def impulse_response_numerical_data(
    system, prec=..., lower_limit=..., upper_limit=..., **kwargs
) -> tuple[Any, Any] | tuple[list[Any], list[Any]]: ...
def impulse_response_plot(
    system, color=..., prec=..., lower_limit=..., upper_limit=..., show_axes=..., grid=..., show=..., **kwargs
) -> None: ...
def ramp_response_numerical_data(
    system, slope=..., prec=..., lower_limit=..., upper_limit=..., **kwargs
) -> tuple[Any, Any] | tuple[list[Any], list[Any]]: ...
def ramp_response_plot(
    system, slope=..., color=..., prec=..., lower_limit=..., upper_limit=..., show_axes=..., grid=..., show=..., **kwargs
) -> None: ...
def bode_magnitude_numerical_data(
    system, initial_exp=..., final_exp=..., freq_unit=..., **kwargs
) -> tuple[Any | list[Any], Any | list[Any]]: ...
def bode_magnitude_plot(
    system, initial_exp=..., final_exp=..., color=..., show_axes=..., grid=..., show=..., freq_unit=..., **kwargs
) -> None: ...
def bode_phase_numerical_data(
    system, initial_exp=..., final_exp=..., freq_unit=..., phase_unit=..., **kwargs
) -> tuple[Any | list[Any], Any | list[Any]]: ...
def bode_phase_plot(
    system, initial_exp=..., final_exp=..., color=..., show_axes=..., grid=..., show=..., freq_unit=..., phase_unit=..., **kwargs
) -> None: ...
def bode_plot(
    system, initial_exp=..., final_exp=..., grid=..., show_axes=..., show=..., freq_unit=..., phase_unit=..., **kwargs
) -> None: ...
