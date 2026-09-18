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
}

function formatText(text) {
  return text
    .split(/\n{2,}/)
    .map((paragraph) => `<p>${paragraph.trim().replace(/\n/g, '<br>')}</p>`)
    .join('');
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Request failed.');
  }

  return response.json();
}

function updateSelectedFile(file) {
  selectedFile = file;
  selectedFileName.textContent = file ? file.name : 'No file selected';
}

browseButton.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (event) => {
  const file = event.target.files && event.target.files[0];
  if (file) {
    updateSelectedFile(file);
  }
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
    fileInput.files = event.dataTransfer.files;
  }
});

uploadButton.addEventListener('click', async () => {
  if (!selectedFile) {
    setStatus(uploadStatus, 'Please choose a PDF file first.', 'error');
    return;
  }

  if (!selectedFile.name.toLowerCase().endsWith('.pdf')) {
    setStatus(uploadStatus, 'Only PDF files are allowed.', 'error');
    return;
  }

  uploadButton.disabled = true;
  clearStatus(uploadStatus);
  setStatus(uploadStatus, 'Uploading and indexing PDF...', 'info');

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || 'Upload failed.');
    }

    uploadResult.classList.remove('hidden');
    uploadResult.innerHTML = `
      <div><strong>Filename:</strong> ${data.filename}</div>
      <div><strong>Pages:</strong> ${data.pages}</div>
      <div><strong>Chunks created:</strong> ${data.chunks_created}</div>
    `;
    setStatus(uploadStatus, 'PDF uploaded successfully.', 'success');
    await loadDocuments();
  } catch (error) {
    setStatus(uploadStatus, error.message || 'Upload failed.', 'error');
  } finally {
    uploadButton.disabled = false;
  }
});

async function loadDocuments() {
  try {
    const data = await fetchJson(`${API_BASE}/documents`);
    if (!data.documents || data.documents.length === 0) {
      documentsList.innerHTML = '<div class="doc-item">No indexed documents yet.</div>';
      return;
    }

    documentsList.innerHTML = data.documents.map((doc) => `
      <div class="doc-item">
        <div class="doc-row">
          <div>
            <strong>${doc.filename}</strong>
            <div class="doc-meta">Chunks: ${doc.chunks_count} • Pages: ${doc.pages.join(', ') || 'N/A'}</div>
          </div>
          <div>
            <span class="doc-status">${doc.status}</span>
          </div>
        </div>
        <div style="margin-top: 12px; display: flex; justify-content: flex-end;">
          <button class="delete-btn" data-document-id="${doc.document_id}" type="button">Delete</button>
        </div>
      </div>
    `).join('');

    documentsList.querySelectorAll('.delete-btn').forEach((button) => {
      button.addEventListener('click', async () => {
        const documentId = button.dataset.documentId;
        try {
          await fetchJson(`${API_BASE}/documents/${documentId}`, { method: 'DELETE' });
          setStatus(uploadStatus, 'Document deleted.', 'success');
          await loadDocuments();
        } catch (error) {
          setStatus(uploadStatus, error.message || 'Failed to delete document.', 'error');
        }
      });
    });
  } catch (error) {
    documentsList.innerHTML = '<div class="doc-item">Unable to load documents.</div>';
  }
}

askButton.addEventListener('click', async () => {
  const question = questionInput.value.trim();
  if (!question) {
    setStatus(queryStatus, 'Please enter a question before submitting.', 'error');
    return;
  }

  askButton.disabled = true;
  clearStatus(queryStatus);
  setStatus(queryStatus, 'Searching the document and generating an answer...', 'info');
  answerContent.classList.remove('empty');
  answerContent.innerHTML = 'Generating answer...';
  sourcesList.innerHTML = '';
  sourcesList.classList.remove('empty');

  try {
    const data = await fetchJson(`${API_BASE}/query`, {
      method: 'POST',
      body: JSON.stringify({ query: question, top_k: 4 }),
    });

    answerContent.innerHTML = formatText(data.answer || 'No answer available.');

    if (!data.retrieved_chunks || data.retrieved_chunks.length === 0) {
      sourcesList.innerHTML = '<div class="source-item">No sources were retrieved.</div>';
      setStatus(queryStatus, 'No relevant chunks were found.', 'error');
      return;
    }

    sourcesList.innerHTML = data.retrieved_chunks.map((chunk) => `
      <details class="source-item" open>
        <summary>${chunk.filename} • Page ${chunk.page ?? 'N/A'} • Distance ${chunk.distance}</summary>
        <div class="source-details">${chunk.text}</div>
      </details>
    `).join('');

    setStatus(queryStatus, 'Answer generated successfully.', 'success');
  } catch (error) {
    answerContent.classList.add('empty');
    answerContent.textContent = 'Unable to generate an answer.';
    sourcesList.innerHTML = '<div class="source-item">No sources available.</div>';
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
