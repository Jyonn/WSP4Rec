"""Distinguish public descriptive text from explicit missing-value placeholders."""
import re


_MISSING={
    'no overview found','no overview available','overview not available','overview unavailable',
    'no plot available','no plot found','plot unknown','plot is unknown','plot unavailable',
    'no description available','no description found','description unavailable',
    'no synopsis available','no synopsis found','synopsis unavailable','synopsis unknown',
    'n/a','na','none','null','unknown','not available','tbd','coming soon',
}


def has_public_description(value):
    if not isinstance(value,str) or not value.strip():return False
    normalized=re.sub(r'\s+',' ',value.strip()).casefold().rstrip(' .!?:;')
    return normalized not in _MISSING
