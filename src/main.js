const constraints = {
	video: { width: {ideal: 4000}, facingMode: { exact: "environment" } },
}

if ('mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices) {
	console.log("Let's get this party started")
}

const video = document.querySelector('video')
const shutter = document.querySelector('#shutter')
const canvas = document.querySelector('canvas')
const preview = document.querySelector('.preview')
const buttonbar = document.querySelector('.button-bar')
const submit = document.querySelector('#submit')
const retry = document.querySelector('#retry')

navigator.mediaDevices.getUserMedia(constraints).then(stream => {
	video.srcObject = stream
})

shutter.addEventListener('click', () => {
	console.log('click')
	canvas.width = video.videoWidth
	canvas.height = video.videoHeight

	canvas.getContext('2d').drawImage(video, 0, 0)
	preview.style.backgroundImage = `url(${canvas.toDataURL('image/jpg')})`
	buttonbar.style.display = 'flex'
	shutter.style.display = 'none'
})


const reset = () => {
	canvas.style.display = 'none'
	buttonbar.style.display = 'none'
	shutter.style.display = 'block'
	preview.style.backgroundImage = 'none'
}

retry.addEventListener('click', reset)
submit.addEventListener('click', () => {
	console.log(performance.now(), 'click')
	const base64 = canvas.toDataURL('image/png')
	console.log(performance.now(), 'base64')
	const json = JSON.stringify({
			image: base64,
		})
	console.log(performance.now(), 'json')
	fetch('/process', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: json,
	}).then(res => {
		console.log(res)
	}).catch(console.error)
	reset()
})
