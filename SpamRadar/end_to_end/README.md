  ## How to install
 <pre> pip install -r requirement.txt</pre>
  
  ## Modules
   * spam_reporting.py -- Spam-reporting classifier
   * sms_screenshot_classfier.py -- SMS image classifier
   * extract_text_from_screenshot.py -- OCR
   * end_to_end_result.py -- The end-to-end classifer
  ## How to run
  ```
  # spam-reporting classifier
  from spam_reporting import spam_reporting

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  # text_list_train is the list of spam-reporting texts list which is need to be classified.
  # for example, text_list_train = ["how are you?", "Good morning"]
  sr_train = spam_reporting("bert-base-multilingual-uncased", text_list_train, device)

  # SMS image classifier
  from sms_screenshot_classifier import sms_screenshot_classifier

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  labels = ['non-sms', 'sms']
  # img_list_train is the list of SMS images list which is need to be classified.
  # for example, img_list_train = ["path to your image 1", "path to your image 2"]
  sc_train = sms_screenshot_classifier(img_list_train, labels, device)
  ```
