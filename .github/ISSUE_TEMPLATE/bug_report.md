---
name: Bug report
about: Create a report to help us improve
title: "[BUG]"
labels: ''
assignees: ''

---

## 🐞 Describe the Bug

> [!NOTE]
> **What happened?**
>
> *Example: When using the `GaussianMovingAverage` with a small window size, the filter returns NaN values on the first few inputs instead of handling edge cases gracefully.*



---

## 🔁 Steps to Reproduce

> [!NOTE]
> **How can we reproduce the issue?**
> 
> *Example:*
> 1. *Create a new `GaussianMovingAverage(window_size=2)`*
> 2. *Call `.add_and_get(1.0)`*
> 3. *Call `.add_and_get(0.0)`*
> 4. *Observe the output is `NaN` instead of a smoothed value*

1. 

---

## ✅ Expected Behavior

> [!NOTE]
> **What did you expect to happen instead?**
> 
> *Example: The filter should return a value even with fewer samples than the window size. Ideally a weighted average over available data.*



---

## 💻 Environment Info

**OS**, please provide the version number.
- [ ] Windows {VERSION}
- [ ] maxOS {VERSION}
- [ ] Linux {DISTR} {VERSION}

**Python version**, please provide the minor version
- [ ] 3.10.
- [ ] 3.11.
- [ ] 3.12.
- [ ] 3.13.
- [ ] 3.14.

**Library Version**
- 

> [!IMPORTANT]
> You can get this info using:
> ```bash
> python --version
> pip show smoothiepy
> ```

---

## 📷 Screenshots or Output (Optional)

> [!NOTE]
> **Any stack traces, logs, or screenshots**
> 
> *Example:* ```python
> RuntimeWarning: invalid value encountered in scalar divide
> return weighted_sum / cur_weights_sum
> ```

```
# Put any logs / stack traces here
```


> [!CAUTION]
> If your output includes sensitive data, consider redacting it before sharing.

---

## 🧩 Additional Context (Optional)

> [!NOTE]
> **Anything else that helps explain the problem.**
> 
> *Example: This only happens when using `build()` followed by processing a list of length < window size.*
