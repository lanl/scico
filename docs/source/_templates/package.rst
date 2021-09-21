API Reference
=============

.. automodule:: {{ fullname }}

   {% block modules %}
   {% if modules %}
   .. autosummary::
     :toctree:
     :recursive:
   {% for item in modules %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
