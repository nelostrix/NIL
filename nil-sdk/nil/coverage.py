"""Passive, versioned workflow metadata. No backend or hardware probing."""
from copy import deepcopy
from functools import lru_cache
from importlib import metadata, resources
import json


@lru_cache(maxsize=1)
def _inventory():
    value = json.loads(resources.files('nil').joinpath('_workflows.json').read_text(encoding='utf-8'))
    if value.get('schema_version') != 1:
        raise ValueError('Unsupported workflow matrix schema')
    return value


def matrix():
    """Independent inventory snapshot; recorded validation is historical evidence."""
    return deepcopy(_inventory())


def list_workflows(*, area=None, implementation=None, validation=None):
    """Filter the bounded inventory, without claiming complete area coverage."""
    if area is not None and (type(area) is not int or not 1 <= area <= 32):
        raise ValueError('area must be an integer from 1 through 32')
    if implementation is not None and implementation not in {'native', 'integration', 'hybrid', 'unimplemented'}:
        raise ValueError('Unknown implementation kind')
    if validation is not None and validation not in {'validated', 'target_blocked', 'experimental'}:
        raise ValueError('Unknown validation status')
    return deepcopy([row for row in _inventory()['workflows']
        if (area is None or area in row['area_ids'])
        and (implementation is None or implementation == row['implementation'])
        and (validation is None or validation == row['validation'])])


def describe(workflow_id):
    """Return one workflow's declared inputs, outputs, limits and evidence."""
    for row in _inventory()['workflows']:
        if row['id'] == workflow_id:
            return deepcopy(row)
    raise ValueError(f'Unknown workflow: {workflow_id!r}')


def availability(workflow_id):
    """Inspect distribution metadata only; installed packages do not prove readiness."""
    row = describe(workflow_id)
    versions, missing = {}, []
    for name in row['requires']:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    return {'workflow_id': workflow_id,
            'dependency_state': 'missing' if missing else 'detected' if versions else 'not_required',
            'versions': versions, 'missing': missing,
            'runtime_verified': False, 'hardware_verified': False,
            'prerequisites_verified': False}


def areas():
    """All roadmap areas, labelled partial or not_audited; never an overall score."""
    result = deepcopy(_inventory()['areas'])
    for area in result:
        area['workflow_ids'] = [row['id'] for row in _inventory()['workflows'] if area['id'] in row['area_ids']]
        area['inventory_status'] = 'partial' if area['workflow_ids'] else 'not_audited'
    return result
