(function() {

	const scripts = document.getElementsByTagName('script');
	const currentScript = Array.from(scripts).find(script => script.src.includes('chatbot-embed.js'));
	const BASE_URL = currentScript.getAttribute('src').replace('chatbot-embed.js', '');

    function injectChatbotCSS() {
        const styleElement = document.createElement('style');
        styleElement.textContent = `

		:root {
			--border-rad-lg-ai__chatbot__embed__html__custom_css_code : 15px;
			--light-text-ai__chatbot__embed__html__custom_css_code : #fefefe;
		}

		.ai__chatbot__embed__html__custom_css_code *, *::before, *::after {
			box-sizing: border-box !important;
		}

		.ai__chatbot__embed__html__custom_css_code main {
			display: flex;
			align-items: center !important;
			justify-content: center !important;
		}

		.ai__chatbot__embed__html__custom_css_code .chatbot-container > * {
			font-family: arial !important;
			padding: .5em !important;   
		}

		.ai__chatbot__embed__html__custom_css_code .chatbot-header {
			display: flex;
			gap: .6em !important;
		}

		.ai__chatbot__embed__html__custom_css_code .logo {
			width: 75px !important;
		}

		.ai__chatbot__embed__html__custom_css_code .chatbot-conversation-container {
			height: 250px !important;
			overflow-y: scroll !important;
			margin: 1em 0 !important;
		}

		/* stop ugly scroll bar on some browsers */
		.ai__chatbot__embed__html__custom_css_code .chatbot-conversation-container::-webkit-scrollbar {
			display: none;
		}

		.ai__chatbot__embed__html__custom_css_code .chatbot-conversation-container::-moz-scrollbar {
			display: none;
		}

		.ai__chatbot__embed__html__custom_css_code .speech {
			padding: 1em !important;
			max-width: 240px !important; 
			color: var(--light-text-ai__chatbot__embed__html__custom_css_code ) !important;
			min-width: 90% !important;
			border-radius: var(--border-rad-lg-ai__chatbot__embed__html__custom_css_code ) !important; 
			font-size: 1.07em !important;
		}

		.ai__chatbot__embed__html__custom_css_code .speech:first-child {
			margin-top: 0 !important;
		}

		.ai__chatbot__embed__html__custom_css_code .speech-ai {
			background: #334959 !important;
			border-top-left-radius: 0 !important;
			margin: 1.2em 1em 0 0 !important; 
		}

		.ai__chatbot__embed__html__custom_css_code .speech-human {
			margin: 1.2em 0 0 1em !important; 
			background: #2f4f4f !important;
			border-top-right-radius: 0 !important; 
		}

		.ai__chatbot__embed__html__custom_css_code .chatbot-input-container {
			display: flex;
			margin: 0 !important;
			padding: 0.5em !important;
		}
		
		ai__chatbot__embed__html__custom_css_code form {
			margin: 0 !important;
			padding: 0 !important;	
		}

		.ai__chatbot__embed__html__custom_css_code input[type="text"], .ai__chatbot__embed__html__custom_css_code button {
			background-color: transparent !important;
			border-radius: var(--border-rad-lg-ai__chatbot__embed__html__custom_css_code ) !important;
			padding: 1em !important;
			outline:none !important;
			border: 1px solid #6c757d !important;
		}

		.ai__chatbot__embed__html__custom_css_code input[type="text"] {
			color: var(--light-text-ai__chatbot__embed__html__custom_css_code ) !important;
			width: 100% !important;
			border-right: 0 !important; 
			border-top-right-radius: 0 !important;
			border-bottom-right-radius: 0 !important;
			font-size: 1em !important;
			outline:none !important;
			margin: 0 !important;
			padding: 1em !important;
		}

		.ai__chatbot__embed__html__custom_css_code, .ai__chatbot__embed__html__custom_css_code button {
			border-left: 0 !important; 
			border-top-left-radius: 0 !important;
			border-bottom-left-radius: 0 !important;
		}

		.ai__chatbot__embed__html__custom_css_code .send-btn-icon {
			width: 20px !important;
			display: block;
		}

		.ai__chatbot__embed__html__custom_css_code .sub-heading {
			color: var(--light-text-ai__chatbot__embed__html__custom_css_code ) !important;
			font-size: 14px !important;
			margin: 12px 0 0 0 !important;
		}

		.ai__chatbot__embed__html__custom_css_code .sub-heading .name{
			font-size: 14px !important;
			text-transform: uppercase !important;
			font-weight:bold !important;
		}

		@keyframes chatbot_hr_animation {
			0% {
				background-position: 0% 50%;
			}
			50% {
				background-position: 100% 50%;
			}
			100% {
				background-position: 0% 50%;
			}
		}

		.ai__chatbot__embed__html__custom_css_code .chatbot-container hr {
			padding: .1em !important;   
		}

		.ai__chatbot__embed__html__custom_css_code hr.colorful {
			border: none !important;
			height:2px !important;
			opacity:1 !important;
			margin:0 !important;
			padding:0 !important;
			background: linear-gradient(270deg, #C70039,  #FF5733,  #FFC300,  #1E7F00,  #0057FF,  #4A148C,  #8E44AD);
			background-size: 200% 200% !important;
			animation: chatbot_hr_animation 4s ease infinite !important;
		}

		@keyframes chatbot_spin_animation {
			0% { transform: rotate(0deg)  }
			100% { transform: rotate(360deg) }
		}

		.ai__chatbot__embed__html__custom_css_code .overlay {
			position: absolute !important;
			width:100% !important;
			height:100% !important;
			top: 0 !important;
			left: 0 !important;
			right: 0 !important;
			bottom: 0 !important;
			background-color: rgba(0, 0, 0, 0.8) !important;
			display: flex;
			flex-direction: column !important;
			justify-content: center !important;
			align-items: center !important;
			z-index: 1000 !important;
		}

		.ai__chatbot__embed__html__custom_css_code .loading-img {
			width: 50px !important;
			height: 50px !important;
			animation: chatbot_spin_animation 2s linear infinite !important;
		}

		.ai__chatbot__embed__html__custom_css_code .loading-text {
			color: white !important;
			margin-top: 20px !important;
			font-family: 'Poppins', arial !important;
			font-size: 16px !important;
		}

		.ai__chatbot__embed__html__custom_css_code #chatbot-minimized-view img {
			width: 30px !important;
			margin-right: 10px !important;
		}

		.ai__chatbot__embed__html__custom_css_code #chatbot-minimized-view span {
			font-size: 14px !important;
			font-weight:bold !important;
			padding-right:0.3em !important;
		}

		.ai__chatbot__embed__html__custom_css_code #chatbot-minimized-view {
			position: fixed !important;
			bottom: 5px !important;
			right: 5px !important;
			font-family:arial !important;
			background-color: #333 !important;
			color:#eee !important;
			padding: 10px !important;
			border-radius: 50px !important;
			box-shadow: 0 2px 10px rgba(0,0,0,0.8) !important;
			cursor: pointer !important;
			display: none;
			align-items: center !important;
			z-index: 1000 !important;
		}

		.ai__chatbot__embed__html__custom_css_code #chatbot-minimized-view img {
			width: 30px !important;
			margin-right: 10px !important;
		}

		.ai__chatbot__embed__html__custom_css_code #chatbot-minimized-view span {
			font-size: 14px !important;
		}		
		
		.ai__chatbot__embed__html__custom_css_code #chatbot-minimized-view,
		.ai__chatbot__embed__html__custom_css_code .chatbot-container {
			position: fixed;
			bottom: 20px !important;
			right: 20px !important;
			z-index: 1000 !important;
		}

		.ai__chatbot__embed__html__custom_css_code .chatbot-container {
			position: fixed !important;
			bottom: 5px !important;
			right: 5px !important;
			overflow-y: auto !important;
			z-index: 1000 !important;
			display: none;
			width: 350px !important;
			max-height: 80vh !important;
			background-color: #171f26 !important;
			border-radius: var(--border-rad-lg-ai__chatbot__embed__html__custom_css_code) !important;
			box-shadow: 0 2px 10px rgba(0,0,0,0.3) !important;
		}

        `;

        document.head.appendChild(styleElement);
    }

    function injectChatbotHTML() {
        const chatbotHTML = `
			<div class="ai__chatbot__embed__html__custom_css_code">
				<div id="chatbot-minimized-view" style="display:flex;">
					<img src="${BASE_URL}images/logo.png" alt="Chatbot Logo">
					<span>Have a question? Let's Chat!</span>
				</div>			
				<main>    
					<section class="chatbot-container">
						<div class="overlay" id="loading">
							<img src="${BASE_URL}images/logo.png" class="loading-img" alt="Loading...">
							<div class="loading-text" id="loading-text"></div>
						</div>
						<div class="chatbot-header">
							<button id="minimize-btn" style="position: absolute; right: 10px; top: 5px; color:white; border:none !important; cursor:pointer;">X</button>
							<img src="${BASE_URL}images/logo.png" class="logo">
							<span class="sub-heading">
								<span class="name">eTeam AI ChatBot</span>
								<p>How can I help you?</p>
							</span>
						</div>
						<hr class="colorful" />
						<div class="chatbot-conversation-container" id="chatbot-conversation-container"></div>
						<form id="form" class="chatbot-input-container">
							<input name="user-input" type="text" id="user-input" required>
							<button id="submit-btn" class="submit-btn">
								<img src="${BASE_URL}images/send.png" class="send-btn-icon">
							</button>
						</form>
					</section>
				</main>
			</div>
        `;
		
        document.body.insertAdjacentHTML('beforeEnd', chatbotHTML);
    }

	function loadChatbotScript() {
		const scriptElement = document.createElement('script');
		
		if (BASE_URL !== '/') {
			scriptElement.src = `${BASE_URL}/dist/bundle.js?url=${encodeURIComponent(BASE_URL)}`;
		}else {
			scriptElement.src = `./dist/bundle.js`;
		}
		
		scriptElement.async = false;
		document.body.appendChild(scriptElement);
		

		scriptElement.onload = function() {
			console.log('Chatbot script loaded successfully.');
		};

		scriptElement.onerror = function() {
			console.error('Failed to load the chatbot script.');
		};
	}

	function showChatbot() {
		document.querySelector('.chatbot-container').style.display = 'block';
		document.getElementById('chatbot-minimized-view').style.display = 'none';
	}

	function hideChatbot() {
		document.querySelector('.chatbot-container').style.display = 'none';
		document.getElementById('chatbot-minimized-view').style.display = 'flex';
	}
	
	function initializeChatbot() {
		const minimizedView = document.getElementById('chatbot-minimized-view');
		const minimizeBtn = document.getElementById('minimize-btn');

		if (minimizedView) {
			minimizedView.addEventListener('click', showChatbot);
		}

		if (minimizeBtn) {
			minimizeBtn.addEventListener('click', hideChatbot);
		}
	}

    injectChatbotCSS();
    injectChatbotHTML();
    loadChatbotScript();
    
    document.addEventListener('DOMContentLoaded', initializeChatbot);
})();
