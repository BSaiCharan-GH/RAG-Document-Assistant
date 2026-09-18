const API_BASE = window.location.origin;

const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const browseButton = document.getElementById('browseButton');
const uploadButton = document.getElementById('uploadButton');
const selectedFileName = document.getElementById('selectedFileName');
const uploadStatus = document.getElementById('uploadStatus');
const uploadResult = document.getElementById('uploadResult');
const documentsList = document.getElementById('documentsList');
const refreshDocsButton = document.getElementById('refreshDocsButton');
const questionInput = document.getElementById('questionInput');
const askButton = document.getElementById('askButton');
const queryStatus = document.getElementById('queryStatus');
const answerContent = document.getElementById('answerContent');
const sourcesList = document.getElementById('sourcesList');

let selectedFile = null;

function setStatus(element, message, type) {
  element.textContent = message;
  element.className = `status-message ${type}`;
  element.classList.remove('hidden');
}

function clearStatus(element) {
  element.className = 'status-message hidden';
  element.textContent = '';
}

function updateSelectedFile(file) {
  selectedFile = file;
  selectedFileName.textContent = file ? file.name : 'No file selected';
}

function clearChildren(element) {
  while (element.firstChild) element.removeChild(element.firstChild);
}

function addLabelValue(parent, label, value) {
  const row = document.createElement('div');
  const strong = document.createElement('strong');
  strong.textContent = `${label}: `;
  row.append(strong, document.createTextNode(String(value)));
  parent.appendChild(row);
}

function renderAnswer(text) {
  clearChildren(answerContent);
  answerContent.classList.remove('empty');

  const paragraphs = String(text || 'No answer available.')
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean);

  if (!paragraphs.length) {
    answerContent.textContent = 'No answer available.';
    return;
  }

  paragraphs.forEach((paragraph) => {
    const p = document.createElement('p');
    p.textContent = paragraph;
    answerContent.appendChild(p);
  });
}

async function fetchJson(url, options = {}) {
  const fetchOptions = { ...options, headers: { ...(options.headers || {}) } };
  if (fetchOptions.body && !(fetchOptions.body instanceof FormData)) {
    fetchOptions.headers['Content-Type'] = 'application/json';
  }

  const response = await fetch(url, fetchOptions);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || 'Request failed.');
  }
  return data;
}

browseButton.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener('change', (event) => {
  const file = event.target.files && event.target.files[0];
  if (file) updateSelectedFile(file);
});

['dragenter', 'dragover'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove('dragover');
  });
});

dropZone.addEventListener('drop', (event) => {
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  if (file) {
    updateSelectedFile(file);
    try {
      fileInput.files = event.dataTransfer.files;
    } catch (_) {
      // The selectedFile state is enough for the upload request.
    }
  }
});

uploadButton.addEventListener('click', async () => {
  if (!selectedFile) {
    setStatus(uploadStatus, 'Please choose a PDF file first.', 'error');
    return;
  }

  if (selectedFile.type && selectedFile.type !== 'application/pdf' && !selectedFile.name.toLowerCase().endsWith('.pdf')) {
    setStatus(uploadStatus, 'Only PDF files are allowed.', 'error');
    return;
  }

  uploadButton.disabled = true;
  clearStatus(uploadStatus);
  setStatus(uploadStatus, 'Uploading and indexing PDF...', 'info');
  uploadResult.classList.add('hidden');

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const data = await fetchJson(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData,
    });

    clearChildren(uploadResult);
    addLabelValue(uploadResult, 'Filename', data.filename);
    addLabelValue(uploadResult, 'Pages', data.pages);
    addLabelValue(uploadResult, 'Chunks created', data.chunks_created);
    uploadResult.classList.remove('hidden');
    setStatus(uploadStatus, 'PDF uploaded and indexed successfully.', 'success');
    updateSelectedFile(null);
    fileInput.value = '';
    await loadDocuments();
  } catch (error) {
    setStatus(uploadStatus, error.message || 'Upload failed.', 'error');
  } finally {
    uploadButton.disabled = false;
  }
});

async function loadDocuments() {
  refreshDocsButton.disabled = true;
  try {
    const data = await fetchJson(`${API_BASE}/documents`);
    clearChildren(documentsList);

    if (!data.documents || data.documents.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'doc-item';
      empty.textContent = 'No indexed documents yet.';
      documentsList.appendChild(empty);
      return;
    }

    data.documents.forEach((doc) => {
      const item = document.createElement('div');
      item.className = 'doc-item';

      const row = document.createElement('div');
      row.className = 'doc-row';

      const details = document.createElement('div');
      const filename = document.createElement('strong');
      filename.textContent = doc.filename;
      const meta = document.createElement('div');
      meta.className = 'doc-meta';
      meta.textContent = `Chunks: ${doc.chunks_count} • Pages: ${doc.pages.join(', ') || 'N/A'}`;
      details.append(filename, meta);

      const status = document.createElement('span');
      status.className = 'doc-status';
      status.textContent = doc.status;
      row.append(details, status);

      const actions = document.createElement('div');
      actions.className = 'doc-actions';
      const deleteButton = document.createElement('button');
      deleteButton.className = 'delete-btn';
      deleteButton.type = 'button';
      deleteButton.textContent = 'Delete';
      deleteButton.dataset.documentId = doc.document_id;
      deleteButton.addEventListener('click', () => deleteDocument(doc.document_id));
      actions.appendChild(deleteButton);

      item.append(row, actions);
      documentsList.appendChild(item);
    });
  } catch (error) {
    clearChildren(documentsList);
    const message = document.createElement('div');
    message.className = 'doc-item';
    message.textContent = error.message || 'Unable to load documents.';
    documentsList.appendChild(message);
  } finally {
    refreshDocsButton.disabled = false;
  }
}

async function deleteDocument(documentId) {
  try {
    await fetchJson(`${API_BASE}/documents/${encodeURIComponent(documentId)}`, { method: 'DELETE' });
    setStatus(uploadStatus, 'Document deleted.', 'success');
    await loadDocuments();
  } catch (error) {
    setStatus(uploadStatus, error.message || 'Failed to delete document.', 'error');
  }
}

function renderSources(chunks) {
  clearChildren(sourcesList);
  sourcesList.classList.remove('empty');

  if (!chunks || chunks.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'source-item';
    empty.textContent = 'No sources were retrieved.';
    sourcesList.appendChild(empty);
    return;
  }

  chunks.forEach((chunk) => {
    const details = document.createElement('details');
    details.className = 'source-item';
    details.open = true;

    const summary = document.createElement('summary');
    summary.textContent = `${chunk.filename} • Page ${chunk.page ?? 'N/A'} • Distance ${chunk.distance}`;

    const sourceText = document.createElement('div');
    sourceText.className = 'source-details';
    sourceText.textContent = chunk.text || '';

    details.append(summary, sourceText);
    sourcesList.appendChild(details);
  });
}

askButton.addEventListener('click', async () => {
  const question = questionInput.value.trim();
  if (!question) {
    setStatus(queryStatus, 'Please enter a question before submitting.', 'error');
    return;
  }

  askButton.disabled = true;
  clearStatus(queryStatus);
  setStatus(queryStatus, 'Searching the documents and generating an answer...', 'info');
  clearChildren(answerContent);
  answerContent.classList.remove('empty');
  answerContent.textContent = 'Generating answer...';
  renderSources([]);

  try {
    const data = await fetchJson(`${API_BASE}/query`, {
      method: 'POST',
      body: JSON.stringify({ query: question, top_k: 4 }),
    });

    renderAnswer(data.answer);
    renderSources(data.retrieved_chunks);
    setStatus(queryStatus, 'Answer generated successfully.', 'success');
  } catch (error) {
    answerContent.classList.add('empty');
    answerContent.textContent = 'Unable to generate an answer.';
    renderSources([]);
    setStatus(queryStatus, error.message || 'Query failed.', 'error');
  } finally {
    askButton.disabled = false;
  }
});

questionInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    askButton.click();
  }
});

refreshDocsButton.addEventListener('click', loadDocuments);
loadDocuments();
