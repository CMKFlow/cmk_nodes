"""CMK contract validation helpers.

The validator checks module contracts at pipe boundaries. It intentionally
reports missing contract fields, not workflow advice.
"""


class CMKValidationError(Exception):
    """Raised when a CMK pipe/module contract is incomplete."""


def _is_missing(value):
    return value is None


def _format_contract_error(node_name, contract_name, required, missing):
    required_lines = "\n".join(f"• {key}" for key in required)
    missing_lines = "\n".join(f"• {key}" for key in missing)
    return (
        f"{node_name}\n\n"
        f"{contract_name} contract incomplete.\n\n"
        f"Required:\n{required_lines}\n\n"
        f"Missing:\n{missing_lines}"
    )


def ensure_pipe(data, node_name="CMK Node", contract_name="Pipe"):
    """Return *data* if it is a dict-like pipe, otherwise raise a CMK error."""
    if data is None:
        raise CMKValidationError(
            f"{node_name}\n\n{contract_name} contract incomplete.\n\nMissing:\n• pipe"
        )
    if not isinstance(data, dict):
        raise CMKValidationError(
            f"{node_name}\n\n{contract_name} contract invalid.\n\n"
            f"Expected pipe dict, got {type(data).__name__}."
        )
    return data


def validate_contract(node_name, contract_name, data, required):
    """Validate that all required contract fields are present and non-None.

    Parameters
    ----------
    node_name: str
        Display name of the node that checks the contract.
    contract_name: str
        Human-readable contract name, for example "Face Source".
    data: dict
        Contract payload to validate.
    required: list[str]
        Required field names.

    Returns
    -------
    dict
        The original data dict for direct use by the caller.
    """
    data = ensure_pipe(data, node_name=node_name, contract_name=contract_name)
    missing = [key for key in required if key not in data or _is_missing(data.get(key))]
    if missing:
        raise CMKValidationError(_format_contract_error(node_name, contract_name, required, missing))
    return data


def validate_values(node_name, contract_name, values, required):
    """Validate an explicit values dict such as direct Create-node inputs."""
    if not isinstance(values, dict):
        raise CMKValidationError(
            f"{node_name}\n\n{contract_name} contract invalid.\n\n"
            f"Expected values dict, got {type(values).__name__}."
        )
    missing = [key for key in required if key not in values or _is_missing(values.get(key))]
    if missing:
        raise CMKValidationError(_format_contract_error(node_name, contract_name, required, missing))
    return values
