# Interview notes — Preprocessing pipelines

## Q1. How do you keep preprocessing from leaking test information?

**60-second answer.** I separate fit from transform. Every statistic that preprocessing needs, such as an
imputed median, a mean and scale, or a list of categories, is learned from the training rows and then
applied unchanged to validation and test rows. I put the steps in one pipeline object and refit that object
inside each cross-validation fold, so the held-out fold never shapes its own inputs. The common leaks are
statistics computed over the full table and categories read from the test table.
**Follow-ups.**
- *Why is a pipeline safer than a loose sequence of transforms?* It is one object with one fit, so the
  validation loop cannot skip a step or refit one on the wrong data.
- *What about target encoding?* It uses the labels, so it must be fitted out of fold, or it leaks the target.

## Q2. What should the model do with a category it has never seen?

**60-second answer.** It depends on the question. With one-hot encoding, an all-zero block says that no
training category applies, which is a sensible default, but it hides how often this happens. I would count
unseen values on the validation data, and if they are frequent, I would add an explicit "other" bucket
learned from training frequencies. The mapping has to be fixed at fit time, so serving data cannot create
new columns.
**Follow-ups.**
- *Why not add a column for the new value?* The number of columns would change, and the trained weights
  would no longer line up with the features.
- *How do you handle many rare categories?* Group the rare ones into an "other" level chosen on training
  data, or use hashing with a stated number of buckets.

## Common mistakes

- Filling missing values with the mean of the full table before splitting.
- Building one-hot columns from the union of train and test categories.
- Using `get_dummies` on a concatenated frame, which makes the test columns depend on the test table.
- Fitting a scaler on the test table "to be consistent".
- Changing the column order between training and serving.
