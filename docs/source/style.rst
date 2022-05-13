.. _scico_dev_style:


Style Guide
===========

.. raw:: html

    <style type='text/css'>
    div.document ul blockquote {
       margin-bottom: 8px !important;
    }
    div.document li > p {
       margin-bottom: 4px !important;
    }
    div.document li {
      list-style: square outside !important;
      margin-left: 1em !important;
    }
    section {
      padding-bottom: 1em;
    }
    ul {
      margin-bottom: 1em;
    }
    </style>


Overview
--------

We adhere to `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ with the exception of allowing a line length limit of 99 characters (as opposed to 79 characters). The standard limit of 72 characters for "flowing long blocks of text" in docstrings or comments is retained. We use `Black <https://github.com/psf/black>`_ as our PEP-8 Formatter and `isort <https://pypi.org/project/isort/>`_ to sort imports. (Please set up a `pre-commit hook <https://pre-commit.com>`_ to ensure any modified code passes format check before it is committed to the development repo.)

We aim to incorporate `PEP 526 <https://www.python.org/dev/peps/pep-0484/>`_ type annotations throughout the library. See the `Mypy <https://mypy.readthedocs.io/en/stable/>`_ type annotation `cheat sheet <https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html>`_ for usage examples. Custom types are defined in :mod:`.typing`.

Our coding conventions are based on both the `NumPy conventions <https://numpydoc.readthedocs.io/en/latest/format.html#overview>`_ and the `Google docstring conventions <https://google.github.io/styleguide/pyguide.html>`_.

Unicode variable names are allowed for internal usage (e.g. for Greek characters for mathematical symbols), but not as part of the public interface for functions or methods.


Naming
------

We follow the `Google naming conventions <https://google.github.io/styleguide/pyguide.html#3164-guidelines-derived-from-guidos-recommendations>`_ listed here:

.. list-table:: Naming Conventions
   :widths: 20 20
   :header-rows: 1

   * - Component
     - Naming Convention
   * - Modules
     - module_name
   * - Package
     - package_name
   * - Class
     - ClassName
   * - Method
     - method_name
   * - Function
     - function_name
   * - Exception
     - ExceptionName
   * - Variable
     - var_name
   * - Parameter
     - parameter_name
   * - Constant
     - CONSTANT_NAME

These names should be descriptive and unambiguous to avoid confusion within the code and other modules in the future.

Example:

.. code:: Python

    d = 6  # Day of the week == Saturday
    if d < 5:
        print("Weekday")

Here the code could be hard to follow since the name ``d`` is not descriptive and requires extra comments to explain the code, which would have been solved otherwise by good naming conventions.

Example:

.. code:: Python

    fldln = 5 # field length

This could be improved by using the descriptive variable ``field_len``.

Things to avoid:

- Single character names except for the following special cases:

   - counters or iterators (``i``, ``j``);
   - `e` as an exception identifier (``Exception e``);
   - `f` as a file in ``with`` statements;
   - mathematical notation in which a reference to the paper or algorithm with said notation is preferred if not clear from the intended purpose.

- Trailing underscores unless the component is meant to be protected or private:

   - protected: Use a single underscore, ``_``, for protected access; and
   - pseudo-private: Use double underscores, ``__``, for pseudo-private access via name mangling.


Displaying and Printing Strings
-------------------------------

We follow the `Google string conventions <https://google.github.io/styleguide/pyguide.html#310-strings>`_. Notably, prefer to use Python f-strings, rather than `.format` or `%` syntax. For example:

.. code:: Python

    state = "active"
    print("The state is %s" % state) # Not preferred
    print(f"The state is {state}")   # Preferred


Imports
-------

We follow the `Google import conventions <https://google.github.io/styleguide/pyguide.html#22-imports>`_. The use of ``import`` statements should be reserved for packages and modules only, i.e. individual classes and functions should not be imported. The only exception to this is the typing module.

-  Use ``import x`` for importing packages and modules, where x is the package or module name.
-  Use ``from x import y`` where x is the package name and y is the module name.
-  Use ``from x import y as z`` if two modules named ``y`` are imported or if ``y`` is too long of a name.
-  Use ``import y as z`` when ``z`` is a standard abbreviation like ``import numpy as np``.


Variables
---------

We follow the `Google variable typing conventions <https://google.github.io/styleguide/pyguide.html#3198-typing-variables>`_ which states that there are a few extra documentation and coding practices that can be applied to variables such as:

- One may type a variables by using a ``: type`` before the function value is assigned, e.g.,

   .. code-block:: python

      a: Foo = SomeDecoratedFunction()

- Avoid global variables.
- A function can refer to variables defined in enclosing functions but cannot assign to them.


Parameters
----------

There are three important style components for parameters inspired by the `NumPy parameter conventions <https://numpydoc.readthedocs.io/en/latest/format.html#parameters>`_:

1. Typing

   We use type annotations meaning we specify the types of the inputs and outputs of any method.
   From the ``typing`` module we can use more types such as ``Optional``, ``Union``, and ``Any``.
   For example,

   .. code-block:: python

      def foo(a: str) -> str:
          """Takes an input of type string and returns a value of type string"""
          ...

2. Default Values

   Parameters should include ``parameter_name = value`` where value is the default for that particular parameter.
   If the parameter has a type then the format is ``parameter_name: Type = value``.
   When documenting parameters, if a parameter can only assume one of a fixed set of values,
   those values can be listed in braces, with the default appearing first.
   For example,

   .. code-block:: python

      """
      letters: {'A', 'B, 'C'}
         Description of `letters`.
      """

3. NoneType

   In Python, ``NoneType`` is a first-class type, meaning the type itself
   can be passed into and returned from functions.
   ``None`` is the most commonly used alias for ``NoneType``.
   If any of the parameters of a function can be ``None`` then it has to be declared.
   ``Optional[T]`` is preferred over ``Union[T, None]``.
   For example,

   .. code-block:: python

      def foo(a: Optional[str], b: Optional[Union[str, int]]) -> str:
      ...

   For documentation purposes, ``NoneType`` or ``None`` should be written with double backticks.


Docstrings
----------

Docstrings are a way to document code within Python and it is the first statement within a package, module, class, or function. To generate a document with all the documentation for the code use `pydoc <https://docs.python.org/3/library/pydoc.html>`_.


Typing
~~~~~~

We follow the `NumPy parameter conventions <https://numpydoc.readthedocs.io/en/latest/format.html#parameters>`_. The following are docstring-specific usages:

- Always enclose variables in single backticks.
- For the parameter types, be as precise as possible, do not use backticks.


Modules
~~~~~~~

We follow the `Google module conventions <https://google.github.io/styleguide/pyguide.html#382-modules>`_. Notably, files must start with a docstring that describes the functionality of the module. For example,

.. code-block:: python

    """A one-line summary of the module must be terminated by a period.

    Leave a blank line and describe the module or program. Optionally describe exported classes, functions, and/or usage
    examples.

    Usage Example:

    foo = ClassFoo()
    bar = foo.FunctionBar()
    """"


Functions
~~~~~~~~~

The word *function* encompasses functions, methods, or generators in this section.
The docstring should give enough information to make calls to the function without needing to read the functions code.

We follow the `Google function conventions <https://google.github.io/styleguide/pyguide.html#383-functions-and-methods>`_. Notably, functions should contain docstrings unless:
- not externally visible (the function name is prefaced with an underscore) or
- very short.

The docstring should be imperative-style ``"""Fetch rows from a Table"""`` instead of the descriptive-style ``"""Fetches rows from a Table"""``. If the method overrides a method from a base class then it may use a simple docstring referencing that base class such as ``"""See base class"""``, unless the behavior is different from the overridden method or there are extra details that need to be documented.

| There are three sections to function docstrings:

- Args:
    - List each parameter by name, and include a description for each parameter.
- Returns: (or Yield in the case of generators)
    - Describe the type of the return value. If a function only returns ``None`` then this section is not required.
- Raises:
   - List all exceptions followed by a description. The name and description should be separated by a colon followed by a space.

Example:

.. code-block:: python

    def fetch_smalltable_rows(table_handle: smalltable.Table,
                              keys: Sequence[Union[bytes, str]],
                              require_all_keys: bool = False,
    ) -> Mapping[bytes, Tuple[str]]:
        """Fetch rows from a Smalltable.

        Retrieve rows pertaining to the given keys from the Table instance
        represented by table_handle. String keys will be UTF-8 encoded.

        Args:
            table_handle:
               An open smalltable.Table instance.
            keys:
               A sequence of strings representing the key of each table
               row to fetch. String `keys` will be UTF-8 encoded.
            require_all_keys: Optional
               If `require_all_keys` is ``True`` only
               rows with values set for all keys will be returned.

        Returns:
            A dict mapping keys to the corresponding table row data
            fetched. Each row is represented as a tuple of strings. For
            example:

            {b'Serak': ('Rigel VII', 'Preparer'),
             b'Zim': ('Irk', 'Invader'),
             b'Lrrr': ('Omicron Persei 8', 'Emperor')}

            Returned keys are always bytes. If a key from the keys argument is
            missing from the dictionary, then that row was not found in the
            table (and require_all_keys must have been False).

        Raises:
            IOError: An error occurred accessing the smalltable.
        """


Classes
~~~~~~~

We follow the `Google class conventions <https://google.github.io/styleguide/pyguide.html#384-classes>`_. Classes, like functions, should have a docstring below the definition describing the class and the class functionality. If the class contains public attributes, the class should have an attributes section where each attribute is listed by name and followed by a description, separated by a colon, like for function parameters. For example,

| Example:

.. code:: Python

    class foo:
	"""One-liner describing the class.

        Additional information or description for the class.
        Can be multi-line

        Attributes:
            attr1: First attribute of the class.
            attr2: Second attribute of the class.
        """

    def __init__(self):
        """Should have a docstring of type function."""
        pass

    def method(self):
        """Should have a docstring of type: function."""
        pass


Extra Sections
~~~~~~~~~~~~~~

We follow the `NumPy style guide <https://numpydoc.readthedocs.io/en/latest/format.html#sections>`_. Notably, the following are sections that can be added to functions, modules, classes, or method definitions.

-  See Also:

   - Refers to related code. Used to direct users to other modules, functions, or classes that they may not be aware of.
   - When referring to functions in the same sub-module, no prefix is needed. Example: For ``numpy.mean`` inside the same sub-module:

     .. code-block:: python

       """
       See Also
       --------
       average: Weighted average.
       """

   - For a reference to ``fft`` in another module:

     .. code-block:: python

       """
       See Also
       --------
       fft.fft2: 2-D fast discrete Fourier transform.
       """

-  Notes

   -  Provides additional information about the code. May include mathematical equations in LaTeX format.
      For example,

     .. code-block:: python

       """
       Notes
       -----
       The FFT is a fast implementation of the discrete Fourier transform:
       .. math::
            X(e^{j\omega } ) = x(n)e^{ - j\omega n}
       """

    | Additionally, math can be used inline:

     .. code-block:: python

       """
       Notes
       -----
       The value of :math:`\omega` is larger than 5.
       """

-  Examples:

   -  Uses the doctest format and is meant to showcase usage.
   -  If there are multiple examples include blank lines before and after each example.
      For example,

     .. code-block:: python

       """
       Examples
       --------
       Necessary imports
       >>> import numpy as np

       Comment explaining example 1.

       >>> np.add(1, 2)
       3

       Comment explaining a new example.

       >>> np.add([1, 2], [3, 4])
       array([4, 6])

       If the example is too long then each line after the first start it
       with a ``...``

       >>> np.add([[1, 2], [3, 4]],
       ...        [[5, 6], [7, 8]])
       array([[ 6,  8],
              [10, 12]])

       """


Comments
~~~~~~~~

There are two types of comments: *block* and *inline*. A good rule of thumb to follow for when to include a comment in your code is *if you have to explain it or is too hard to figure out at first glance, then comment it*.
An example of this, taken from the `Google comment conventions <https://google.github.io/styleguide/pyguide.html#385-block-and-inline-comments>`_, is complicated operations which most likely require a block of comments beforehand.

.. code-block:: Python

    # We use a block comment because the following code performs a
    # difficult operation. Here we can explain the variables or
    # what the concept of the operation does in an easier
    # to understand way.

    i = i & (i-1) == 0:  # true if i is 0 or a power of 2 [explains the concept not the code]

If a comment consists of one or more full sentences (as is typically the case for *block* comments), it should start with an upper case letter and end with a period. *Inline* comments often consist of a brief phrase which is not a full sentence, in which case they should have a lower case initial letter and not have a terminating period.


Markup
~~~~~~

The following components require the recommended markup taken from the `NumPy Conventions <https://numpydoc.readthedocs.io/en/latest/format.html#common-rest-concepts>`__.:

- Paragraphs:
  Indentation is significant and indicates the indentation of the output. New paragraphs are marked with a blank line.
- Variable, parameter, module, function, method, and class names:
  Should be written between single back-ticks (e.g. \`x\`, rendered as `x`), but note that use of `Sphinx cross-reference syntax <https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects>`_ is preferred for modules (`:mod:\`module-name\`` ), functions (`:func:\`function-name\`` ), methods (`:meth:\`method-name\`` ) and classes (`:class:\`class-name\`` ).
- None, NoneType, True, and False:
  Should be written between double back-ticks (e.g. \`\`None\`\`, \`\`True\`\`, rendered as ``None``, ``True``).
- Types:
  Should be written between double back-ticks (e.g. \`\`int\`\`, rendered as ``int``).

Other components can use \*italics\*, \*\*bold\*\*, and \`\`monospace\`\` (respectively rendered as *italics*, **bold**, and ``monospace``) if needed, but not for variable names, doctest code, or multi-line code.


Documentation
-------------

Documentation that is separate from code (like this page)
should follow the
`IEEE Style Manual
<https://journals.ieeeauthorcenter.ieee.org/your-role-in-article-production/ieee-editorial-style-manual/>`_.
For additional grammar and usage guidance,
refer to `The Chicago Manual of Style <https://www.chicagomanualofstyle.org/>`_.
A few notable guidelines:

    * Equations which conclude a sentence should end with a period,
      e.g., "Poisson's equation is

      .. math::

       \Delta \varphi = f \;."

    * Do not capitalize acronyms or inititalisms when defining them,
      e.g., "computer-aided system engineering (CASE),"
      "fast Fourier transform (FFT)."

    * Avoid capitalization in text except where absolutely necessary,
      e.g., "Newton’s first law."

    * Use a single space after the period at the end of a sentence.


The source code (`.rst` files) for these pages does not have a line-length guideline,
but line breaks at or before 79 characters are encouraged.
