---
name: Feature request
about: Suggest an idea for this project
title: "[FEATURE]"
labels: ''
assignees: ''

---

## 🧩 Problem Description

> [!NOTE]
> **What problem are you trying to solve?**
> 
> *Example: I'm using this library for real-time eye tracking and noticed I can't chain 2D filters dynamically. This limits our ability to smooth gaze data effectively during live experiments.*



---

## 🌟 Proposed Solution

> [!NOTE]
> **What should the feature do? Describe your ideal outcome.**
> 
> *Example: I’d like to be able to attach filters dynamically to a `Smoother2D` instance during runtime. Something like `smoother.attach(filter)` without rebuilding everything.*



---

## 🔁 Alternatives or Workarounds (Optional)

> [!NOTE]
> **What have you tried or considered instead?**
> 
> *Example: I tried subclassing `Smoother2D` and overriding `.process()`, but that breaks compatibility with some built-in filters.*



---

## 📎 Related Info (Optional)

> [!NOTE]
> Can include:
> - Screenshots or diagrams
> - API usage examples
> - Performance implications



---

## 📝 Suggested API or Code Snippet (Optional)

> [!NOTE]
> *Example:*
> ```python
> smoother = # ... Build Smoother 
> smoother.attach(GaussianFilter2D(std=0.5))
> smoother.attach(FixationFilter(threshold=0.2))
> smoother.build()
> ```

```python
# Put your code here

```
