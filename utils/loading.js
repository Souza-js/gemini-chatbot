const showLoading = function (text) {
	const loading = document.getElementById('loading')
	const loadingText = document.getElementById('loading-text')
	
	loading.style.display = 'flex'
	loadingText.innerHTML = text || "Thinking..."
}

const hideLoading = function () {
	const loading = document.getElementById('loading')
	loading.style.display = 'none';
}

showLoading("Loading...");

export {showLoading, hideLoading}