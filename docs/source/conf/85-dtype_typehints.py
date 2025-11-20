from typing import Optional, Sequence, Union  # needed for typehints_formatter hack

from scico.typing import (  # needed for typehints_formatter hack
    ArrayIndex,
    AxisIndex,
    DType,
)


# An explanation for this nasty hack, the primary purpose of which is to avoid
# the very long definition of the scico.typing.DType appearing explicitly in the
# docs. This is handled correctly by sphinx.ext.autodoc in some circumstances,
# but only when sphinx_autodoc_typehints is not included in the extension list,
# and the appearance of the type hints (e.g. whether links to definitions are
# included) seems to depend on whether "from __future__ import annotations" was
# used in the module being documented, which is not ideal from a consistency
# perspective. (It's also worth noting that sphinx.ext.autodoc provides some
# configurability for type aliases via the autodoc_type_aliases sphinx
# configuration option.) The alternative is to include sphinx_autodoc_typehints,
# which gives a consistent appearance to the type hints, but the
# autodoc_type_aliases configuration option is ignored, and type aliases are
# always expanded. This hack avoids expansion for the type aliases with the
# longest definitions by definining a custom function for formatting the
# type hints, using an option provided by sphinx_autodoc_typehints. For
# more information, see
#   https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_type_aliases
#   https://github.com/tox-dev/sphinx-autodoc-typehints/issues/284
#   https://github.com/tox-dev/sphinx-autodoc-typehints/blob/main/README.md
def typehints_formatter_function(annotation, config):
    markup = {
        DType: ":obj:`~scico.typing.DType`",
        # Compound types involving DType must be added here to avoid their DType
        # component being expanded in the docs.
        Optional[DType]: r":obj:`~typing.Optional`\ [\ :obj:`~scico.typing.DType`\ ]",
        Union[DType, Sequence[DType]]: (
            r":obj:`~typing.Union`\ [\ :obj:`~scico.typing.DType`\ , "
            r":obj:`~typing.Sequence`\ [\ :obj:`~scico.typing.DType`\ ]]"
        ),
        AxisIndex: ":obj:`~scico.typing.AxisIndex`",
        ArrayIndex: ":obj:`~scico.typing.ArrayIndex`",
    }
    if annotation in markup:
        return markup[annotation]
    else:
        return None


typehints_formatter = typehints_formatter_function
