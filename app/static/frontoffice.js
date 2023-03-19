class FrontOffice {

    constructor(onModelReady, onWsOpened, onRecordingEnabled, onMessageReceived) {
        this.onModelReady = onModelReady;
        this.onWsOpened = onWsOpened;
        this.onRecordingEnabled = onRecordingEnabled;
        this.onMessageReceived = onMessageReceived;
        this.modelReady = false;
        this.websocket = undefined;
        this.mediaRecorder = undefined;
    }

    async init() {
        await this._isModelReady();
        await this._getWebsocket();
        await this._getMediaServices();
    }

    async _isModelReady() {
        if (!this.modelReady) {
            try {
                let response = await fetch('/ready')
                let data = await response.json()
                if (data.ready) {
                    this.modelReady = true;
                    this.onModelReady(data.filter_pipeline_dir);
                } else {
                    this.modelReady = false;
                    this.onModelReady(undefined);
                }
            } catch (error) {
                console.log(error)
                this.modelReady = false;
                this.onModelReady(data.filter_pipeline_dir);
            }
        }
        return this.modelReady;
    }

    async _getWebsocket() {
        if (this.websocket === undefined) {
            this.websocket = new WebSocket(this._buildWsUrl())
            this.websocket.onopen = () => {
                console.log('websocket connection opened')
                this.onWsOpened(true)
            }

            this.websocket.onmessage = (event) => {
                if (event) {
                    const message = JSON.parse(event.data);
                    console.log('websocket message received', message)
                    this.onMessageReceived(message)
                }
            }

            this.websocket.onclose = () => {
                console.log('websocket connection closed')
                this.websocket = undefined;
                this.onWsOpened(false)
            }

            this.websocket.onerror = (error) => {
                console.log('websocket error', error)
            }
        }
        return this.websocket;
    }

    async _getMediaServices() {
        let ready = await this._isModelReady()
        if (!ready) {
            console.log('Model needs to be ready to start recording')
            return
        }
        let socket = await this._getWebsocket()
        if (this.mediaServices === undefined) {
            try {
                this.mediaServices = navigator.mediaDevices;
                if (this.mediaServices) {
                    if (!MediaRecorder.isTypeSupported('audio/webm'))
                        return alert('Browser does not support audio recording.');

                    let stream = await this.mediaServices.getUserMedia({audio: true})
                    this.mediaRecorder = new MediaRecorder(stream, {mimeType: 'audio/webm'})

                    this.mediaRecorder.addEventListener('dataavailable', async (event) => {
                        if (event.data.size > 0 && socket.readyState === 1) {
                            socket.send(event.data)
                        }
                    })
                    this.mediaRecorder.start(250)
                    this.onRecordingEnabled(true)
                }
            } catch (error) {
                console.log(error)
                this.onRecordingEnabled(true)
            }
        }
        return this.mediaServices;
    }

    async sendMessage(message) {
        if (message) {
            try {
                let ready = await this._isModelReady()
                if (!ready) {
                    console.log('Model not ready')
                    return
                }
                let response = await fetch('/text', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({'text': message})
                })
                let data = await response.json()
                for (let d of data) {
                    this.onMessageReceived(d)
                }
            } catch (error) {
                console.log(error)
            }
        }
    }

    _buildWsUrl() {
        const url = new URL(window.location.href);
        const wsProto = (window.location.protocol === "https:") ? "wss://" : "ws://";
        const wsPort = url.port ? ":" + url.port : "";
        const wsUrl = wsProto + url.hostname + wsPort + "/listen";
        return wsUrl;
    }


}
