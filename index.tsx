import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

// --- Global Config & State ---
const CONFIG = {
    particleCount: 1500,
    dustCount: 2500,
    colors: {
        gold: 0xd4af37,
        cream: 0xfceea7,
        red: 0x880000,
        green: 0x004400,
        blue: 0x000088
    }
};

const STATE = {
    mode: 'TREE', // TREE, SCATTER, FOCUS
    focusTarget: null,
    handData: {
        detected: false,
        centerX: 0.5,
        centerY: 0.5
    }
};

// --- Utilities ---
class Utils {
    static createCandyCaneTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');
        if (!ctx) return new THREE.Texture();
        
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, 64, 64);
        
        // Draw diagonal stripes
        ctx.strokeStyle = '#aa0000';
        ctx.lineWidth = 16;
        ctx.beginPath();
        ctx.moveTo(0, 0); ctx.lineTo(64, 64);
        ctx.moveTo(32, -32); ctx.lineTo(96, 32);
        ctx.moveTo(-32, 32); ctx.lineTo(32, 96);
        ctx.stroke();

        const texture = new THREE.CanvasTexture(canvas);
        texture.wrapS = THREE.RepeatWrapping;
        texture.wrapT = THREE.RepeatWrapping;
        texture.repeat.set(1, 4);
        return texture;
    }
}

// --- Particle Class ---
class Particle {
    mesh: THREE.Object3D;
    type: string;
    baseScale: THREE.Vector3;
    targetPos: THREE.Vector3;
    targetRot: THREE.Euler;
    targetScale: THREE.Vector3;
    velocity: THREE.Vector3;

    constructor(mesh: THREE.Object3D, type = 'DECORATION') {
        this.mesh = mesh;
        this.type = type; // 'DECORATION', 'PHOTO', 'DUST'
        this.baseScale = mesh.scale.clone();
        this.targetPos = new THREE.Vector3();
        this.targetRot = new THREE.Euler();
        this.targetScale = new THREE.Vector3();
        
        this.velocity = new THREE.Vector3(
            (Math.random() - 0.5) * 0.1,
            (Math.random() - 0.5) * 0.1,
            (Math.random() - 0.5) * 0.1
        );
        
        // Initial placement
        this.setScatterTarget();
        this.mesh.position.copy(this.targetPos);
    }

    update(dt: number, time: number) {
        // Lerp position
        this.mesh.position.lerp(this.targetPos, 0.05);
        
        // Rotation Logic
        if (STATE.mode === 'SCATTER') {
            // Auto rotation based on velocity vector
            this.mesh.rotation.x += this.velocity.x * 2;
            this.mesh.rotation.y += this.velocity.y * 2;
        } else {
            // Lerp to target rotation
            this.mesh.rotation.x += (this.targetRot.x - this.mesh.rotation.x) * 0.05;
            this.mesh.rotation.y += (this.targetRot.y - this.mesh.rotation.y) * 0.05;
            this.mesh.rotation.z += (this.targetRot.z - this.mesh.rotation.z) * 0.05;
        }

        // Scale Logic
        this.mesh.scale.lerp(this.targetScale, 0.05);
    }

    setTreeTarget(i: number, total: number) {
        const t = i / total;
        const angle = t * 50 * Math.PI;
        const maxRadius = 15;
        const radius = maxRadius * (1 - t);
        const height = t * 40 - 20;

        this.targetPos.set(
            Math.cos(angle) * radius,
            height,
            Math.sin(angle) * radius
        );
        
        // Add some jitter
        this.targetPos.x += (Math.random() - 0.5);
        this.targetPos.z += (Math.random() - 0.5);

        this.targetRot.set(Math.random() * Math.PI, Math.random() * Math.PI, 0);
        this.targetScale.copy(this.baseScale);
    }

    setScatterTarget() {
        // Spherical distribution radius 8~20
        const r = 8 + Math.random() * 12;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos((Math.random() * 2) - 1);

        this.targetPos.set(
            r * Math.sin(phi) * Math.cos(theta),
            r * Math.sin(phi) * Math.sin(theta),
            r * Math.cos(phi)
        );
        
        this.targetScale.copy(this.baseScale);
    }

    setFocusTarget(isFocusItem: boolean) {
        if (isFocusItem) {
            // Place item at local (0, 2, 40). 
            // Combined with MainGroup rotation (0,0,0), this centers it in front of camera (0, 2, 50).
            this.targetPos.set(0, 2, 40); 
            this.targetRot.set(0, 0, 0);
            this.targetScale.set(3, 3, 3);
        } else {
            // Push others back and scatter
            this.setScatterTarget();
            this.targetPos.multiplyScalar(2.0); // Push further out
        }
    }
}

// --- Main Application ---
class App {
    particles: Particle[] = [];
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    composer: EffectComposer;
    mainGroup: THREE.Group;
    dustSystem: THREE.Points | null = null;
    visionManager: VisionManager;
    bgm: HTMLAudioElement | null = null;

    constructor() {
        this.initThree();
        this.initPostProcessing();
        this.initContent();
        this.initEvents();
        
        this.visionManager = new VisionManager(this);
        this.visionManager.start();

        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }

    initThree() {
        this.scene = new THREE.Scene();
        
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(0, 2, 50);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.toneMapping = THREE.ReinhardToneMapping;
        this.renderer.toneMappingExposure = 2.2;
        document.body.appendChild(this.renderer.domElement);

        // Environment
        const pmremGenerator = new THREE.PMREMGenerator(this.renderer);
        this.scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;

        // Lighting
        const ambient = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambient);

        const point = new THREE.PointLight(0xffaa00, 2, 50);
        this.scene.add(point);

        const spotGold = new THREE.SpotLight(CONFIG.colors.gold, 1200);
        spotGold.position.set(30, 40, 40);
        spotGold.angle = 0.5;
        spotGold.penumbra = 0.5;
        this.scene.add(spotGold);

        const spotBlue = new THREE.SpotLight(CONFIG.colors.blue, 600);
        spotBlue.position.set(-30, 20, -30);
        spotBlue.lookAt(0, 0, 0);
        this.scene.add(spotBlue);

        this.mainGroup = new THREE.Group();
        this.scene.add(this.mainGroup);
    }

    initPostProcessing() {
        this.composer = new EffectComposer(this.renderer);
        const renderPass = new RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);

        const bloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            0.45, // strength
            0.4,  // radius
            0.7   // threshold
        );
        this.composer.addPass(bloomPass);
    }

    initContent() {
        // Shared Materials
        const matGold = new THREE.MeshStandardMaterial({ color: CONFIG.colors.gold, roughness: 0.2, metalness: 0.9 });
        const matGreen = new THREE.MeshStandardMaterial({ color: CONFIG.colors.green, roughness: 0.8, metalness: 0.1 });
        const matRed = new THREE.MeshPhysicalMaterial({ color: CONFIG.colors.red, roughness: 0.1, metalness: 0.2, clearcoat: 1.0 });
        const matOrn = new THREE.MeshPhysicalMaterial({ color: CONFIG.colors.gold, roughness: 0.1, metalness: 0.8, clearcoat: 1.0 });

        // Shared Geometries
        const geoBox = new THREE.BoxGeometry(1, 1, 1);
        const geoSphere = new THREE.SphereGeometry(0.6, 32, 32);
        
        // Candy Cane Geo
        const curve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(0, -1, 0),
            new THREE.Vector3(0, 1, 0),
            new THREE.Vector3(0.5, 1.5, 0)
        ]);
        const geoCane = new THREE.TubeGeometry(curve, 20, 0.15, 8, false);
        const texCane = Utils.createCandyCaneTexture();
        const matCane = new THREE.MeshStandardMaterial({ map: texCane, roughness: 0.4 });

        // 1. Main Particles
        for (let i = 0; i < CONFIG.particleCount; i++) {
            let mesh;
            let type = 'DECORATION';
            const rand = Math.random();

            if (rand < 0.1) {
                // Candy Cane
                mesh = new THREE.Mesh(geoCane, matCane);
            } else if (rand < 0.4) {
                // Box Gift (Gold or Green)
                mesh = new THREE.Mesh(geoBox, Math.random() > 0.5 ? matGold : matGreen);
                mesh.scale.setScalar(0.8 + Math.random() * 0.5);
            } else {
                // Sphere Ornament (Red or Gold)
                mesh = new THREE.Mesh(geoSphere, Math.random() > 0.5 ? matRed : matOrn);
            }

            this.mainGroup.add(mesh);
            const p = new Particle(mesh, type);
            p.setTreeTarget(i, CONFIG.particleCount);
            this.particles.push(p);
        }

        // 3. Dust Particles
        const dustGeo = new THREE.BufferGeometry();
        const dustPos = [];
        for(let i=0; i< CONFIG.dustCount; i++) {
            const r = 25;
            dustPos.push((Math.random()-0.5)*r, (Math.random()-0.5)*r, (Math.random()-0.5)*r);
        }
        dustGeo.setAttribute('position', new THREE.Float32BufferAttribute(dustPos, 3));
        const dustMat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.05, transparent: true, opacity: 0.6 });
        const dustSystem = new THREE.Points(dustGeo, dustMat);
        this.scene.add(dustSystem);
        this.dustSystem = dustSystem;
    }

    addPhotoToScene(texture: THREE.Texture) {
        // Photo Frame Logic
        const frameGeo = new THREE.BoxGeometry(2.2, 2.2, 0.2);
        const matFrame = new THREE.MeshStandardMaterial({ color: CONFIG.colors.gold, roughness: 0.3, metalness: 0.8 });
        const matPhoto = new THREE.MeshBasicMaterial({ map: texture });
        
        const frame = new THREE.Mesh(frameGeo, matFrame);
        
        // The photo plane slightly in front
        const photoPlane = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), matPhoto);
        photoPlane.position.z = 0.11;
        frame.add(photoPlane);

        this.mainGroup.add(frame);
        
        const p = new Particle(frame, 'PHOTO');
        // Insert into random position or append
        p.setTreeTarget(Math.floor(Math.random() * CONFIG.particleCount), CONFIG.particleCount);
        this.particles.push(p);
    }

    initEvents() {
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            this.composer.setSize(window.innerWidth, window.innerHeight);
        });

        const toggleUI = () => {
            document.querySelector('#controls-layer')?.classList.toggle('ui-hidden');
        };

        // Keydown 'H'
        window.addEventListener('keydown', (e) => {
            if (e.key === 'h' || e.key === 'H') {
                toggleUI();
            }
        });

        // Button: Start (Enables Vision, Hides UI, Plays Music)
        document.getElementById('btn-start')?.addEventListener('click', () => {
            // Enable gesture recognition
            this.visionManager.isEnabled = true;
            this.updateMode('TREE');
            // Hide UI immediately for immersion
            toggleUI();
            
            // Play Background Music
            if (this.bgm) {
                this.bgm.play().catch(e => console.warn('Music playback failed:', e));
            }
        });

        // Button: Reset (Disables Vision, Resets State, Pauses Music)
        document.getElementById('btn-reset')?.addEventListener('click', () => {
            // Stop gesture recognition
            this.visionManager.isEnabled = false;
            
            // Reset state to initial
            this.mainGroup.rotation.set(0, 0, 0);
            this.updateMode('TREE');
            
            // Clear focus target
            STATE.focusTarget = null;
            STATE.handData.detected = false;

            // Pause Music
            if (this.bgm) {
                this.bgm.pause();
                this.bgm.currentTime = 0;
            }
        });

        // Button: Add Memories (triggers file input)
        document.getElementById('btn-upload')?.addEventListener('click', () => {
             document.getElementById('file-input')?.click();
        });

        // File Upload Change Event (Photos)
        const fileInput = document.getElementById('file-input');
        fileInput?.addEventListener('change', (e: Event) => {
            const target = e.target as HTMLInputElement;
            const f = target.files?.[0];
            if (!f) return;
            
            const reader = new FileReader();
            reader.onload = (ev) => {
                new THREE.TextureLoader().load(ev.target?.result as string, (t) => {
                    t.colorSpace = THREE.SRGBColorSpace; 
                    this.addPhotoToScene(t);
                    // Switch to focus to show the new photo immediately
                    this.updateMode('FOCUS');
                    // Ensure the last added photo is the focus target
                    const photos = this.particles.filter(p => p.type === 'PHOTO');
                    STATE.focusTarget = photos[photos.length - 1];
                    this.particles.forEach(p => p.setFocusTarget(p === STATE.focusTarget));
                });
            }
            reader.readAsDataURL(f);
        });

        // Button: Set Music (triggers audio input)
        document.getElementById('btn-music')?.addEventListener('click', () => {
            document.getElementById('music-input')?.click();
        });

        // Audio Upload Change Event
        const musicInput = document.getElementById('music-input');
        musicInput?.addEventListener('change', (e: Event) => {
            const target = e.target as HTMLInputElement;
            const f = target.files?.[0];
            if (!f) return;
            
            // Clean up previous audio if any
            if (this.bgm) {
                this.bgm.pause();
                URL.revokeObjectURL(this.bgm.src);
            }

            const url = URL.createObjectURL(f);
            this.bgm = new Audio(url);
            this.bgm.loop = true;
            this.bgm.volume = 0.5; // Set a default volume
            
            // Feedback to user (Update button text)
            const btn = document.getElementById('btn-music');
            if (btn) btn.innerText = "Music Ready";
        });
    }

    updateMode(mode: string) {
        if (STATE.mode === mode) return;
        STATE.mode = mode;

        if (mode === 'TREE') {
            this.particles.forEach((p, i) => p.setTreeTarget(i, this.particles.length));
        } else if (mode === 'SCATTER') {
            this.particles.forEach(p => p.setScatterTarget());
        } else if (mode === 'FOCUS') {
            // Pick random photo
            const photos = this.particles.filter(p => p.type === 'PHOTO');
            if (photos.length > 0) {
                // Always pick random when entering focus
                STATE.focusTarget = photos[Math.floor(Math.random() * photos.length)];
                this.particles.forEach(p => p.setFocusTarget(p === STATE.focusTarget));
            } else {
                // Fallback if no photos
                this.updateMode('TREE');
            }
        }
    }

    animate(time: number) {
        requestAnimationFrame(this.animate);
        
        const t = time * 0.001;

        // Update Particles
        this.particles.forEach(p => p.update(0.016, t));

        // Dust Rotation
        if (this.dustSystem) {
            this.dustSystem.rotation.y = t * 0.05;
        }

        // Interaction Mapping
        if (STATE.mode === 'FOCUS') {
            // Force reset rotation to alignment so the focused item (at local 0,2,40) faces camera
            // We damp the rotation back to 0,0,0
            this.mainGroup.rotation.x += (0 - this.mainGroup.rotation.x) * 0.1;
            this.mainGroup.rotation.y += (0 - this.mainGroup.rotation.y) * 0.1;
            this.mainGroup.rotation.z += (0 - this.mainGroup.rotation.z) * 0.1;
        } else if (STATE.handData.detected && this.visionManager.isEnabled) {
            // Map hand center to scene rotation with damping
            const targetRotY = (STATE.handData.centerX - 0.5) * 4; // -2 to 2 rad
            const targetRotX = (STATE.handData.centerY - 0.5) * 2;
            
            this.mainGroup.rotation.y += (targetRotY - this.mainGroup.rotation.y) * 0.1;
            this.mainGroup.rotation.x += (targetRotX - this.mainGroup.rotation.x) * 0.1;
        } else {
            // Idle auto rotation
            this.mainGroup.rotation.y += 0.002;
            this.mainGroup.rotation.x *= 0.95; // Return to level
        }

        this.composer.render();
    }
}

// --- Computer Vision (MediaPipe) ---
class VisionManager {
    app: App;
    video: HTMLVideoElement;
    lastVideoTime: number = -1;
    landmarker: HandLandmarker | null = null;
    isEnabled: boolean = false;

    constructor(app: App) {
        this.app = app;
        this.video = document.getElementById('webcam') as HTMLVideoElement;
    }

    async start() {
        try {
            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
            );
            
            this.landmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
                    delegate: "GPU"
                },
                runningMode: "VIDEO",
                numHands: 1
            });

            // Start Webcam
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            this.video.srcObject = stream;
            this.video.addEventListener("loadeddata", () => {
                this.predictWebcam();
                // Hide Loader
                const loader = document.getElementById('loader');
                if (loader) {
                    loader.style.opacity = '0';
                    setTimeout(() => loader.style.display = 'none', 1000);
                }
            });

        } catch (e) {
            console.error("MediaPipe Error:", e);
            // Allow app to run without camera
            const loader = document.getElementById('loader');
            if (loader) {
                loader.style.opacity = '0';
                setTimeout(() => loader.style.display = 'none', 1000);
            }
        }
    }

    predictWebcam() {
        // Always loop to keep it alive
        requestAnimationFrame(() => this.predictWebcam());

        // Skip processing if disabled
        if (!this.isEnabled) {
            STATE.handData.detected = false;
            return;
        }

        if (this.video.currentTime !== this.lastVideoTime) {
            this.lastVideoTime = this.video.currentTime;
            const startTimeMs = performance.now();
            
            if (this.landmarker) {
                const result = this.landmarker.detectForVideo(this.video, startTimeMs);
                this.processGestures(result);
            }
        }
    }

    processGestures(result: any) {
        if (result.landmarks && result.landmarks.length > 0) {
            const lm = result.landmarks[0]; // First hand

            // Map Hand Center (Landmark 9) for Rotation
            // MediaPipe coords: x left->right (0-1), y top->bottom (0-1)
            STATE.handData.detected = true;
            STATE.handData.centerX = 1 - lm[9].x; // Mirror X
            STATE.handData.centerY = lm[9].y;

            // 1. Pinch (Focus)
            // Thumb Tip (4) vs Index Tip (8)
            const pinchDist = Math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y);
            if (pinchDist < 0.05) {
                this.app.updateMode('FOCUS');
                return;
            }

            // Calculate average distance from tips to wrist (0)
            const tips = [8, 12, 16, 20];
            let totalDist = 0;
            tips.forEach(idx => {
                totalDist += Math.hypot(lm[idx].x - lm[0].x, lm[idx].y - lm[0].y);
            });
            const avgDist = totalDist / 4;

            // 2. Fist (Tree)
            if (avgDist < 0.25) {
                this.app.updateMode('TREE');
            }
            // 3. Open Hand (Scatter)
            else if (avgDist > 0.4) {
                this.app.updateMode('SCATTER');
            }

        } else {
            STATE.handData.detected = false;
        }
    }
}

new App();