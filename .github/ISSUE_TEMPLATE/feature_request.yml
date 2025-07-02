name: "✨ Feature request"
description: "Suggest a new idea or improvement"
title: "[FEATURE]"
labels: [enhancement]
assignees: []

body:
  - type: markdown
    attributes:
      value: |
        ## 🧩 Problem Description

        > [!NOTE]
        > **What problem are you trying to solve?**
        >
        > *Example: I'm using this library for real-time eye tracking and noticed I can't chain 2D filters dynamically. This limits our ability to smooth gaze data effectively during live experiments.*

  - type: textarea
    id: problem-description
    attributes:
      label: Problem description
      placeholder: What problem or limitation does this feature address?
    validations:
      required: true

  - type: markdown
    attributes:
      value: |
        ## 🌟 Proposed Solution

        > [!NOTE]
        > **What should the feature do? Describe your ideal outcome.**
        >
        > *Example: I’d like to be able to attach filters dynamically to a `Smoother2D` instance during runtime. Something like `smoother.attach(filter)` without rebuilding everything.*

  - type: textarea
    id: proposed-solution
    attributes:
      label: Proposed solution
      placeholder: Describe how you'd like the feature to work
    validations:
      required: true

  - type: markdown
    attributes:
      value: |
        ## 🔁 Alternatives or Workarounds (Optional)

        > [!NOTE]
        > **What have you tried or considered instead?**
        >
        > *Example: I tried subclassing `Smoother2D` and overriding `.process()`, but that breaks compatibility with some built-in filters.*

  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives or workarounds
      placeholder: Mention any other approaches or workarounds you've tried

  - type: markdown
    attributes:
      value: |
        ## 📎 Related Info (Optional)

        > [!NOTE]
        > Can include:
        > - Screenshots or diagrams
        > - API usage examples
        > - Performance implications

  - type: textarea
    id: related-info
    attributes:
      label: Related information
      placeholder: Paste any helpful links, screenshots, diagrams, or notes here

  - type: markdown
    attributes:
      value: |
        ## 📝 Suggested API or Code Snippet (Optional)

        > [!NOTE]
        > *Example:*
        > ```python
        > smoother = # ... Build Smoother 
        > smoother.attach(GaussianFilter2D(std=0.5))
        > smoother.attach(FixationFilter(threshold=0.2))
        > smoother.build()
        > ```

  - type: textarea
    id: code-snippet
    attributes:
      label: Optional code snippet or API usage
      render: python
      placeholder: Paste an API example or code mockup here
