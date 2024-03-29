console.log('running that scrpt!')

import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r121/three.module.min.js'
			import { GUI } from 'https://unpkg.com/three@0.121.0/examples/jsm/libs/dat.gui.module.js';
			import { OrbitControls } from 'https://unpkg.com/three@0.121.0/examples/jsm/controls/OrbitControls.js';
			import Stats from 'https://unpkg.com/three@0.121.0/examples/jsm/libs/stats.module.js';
			
			import { PLYLoader } from 'https://unpkg.com/three@0.121.0/examples/jsm/loaders/PLYLoader.js';

			var container, stats;

			var camera, cameraTarget, scene, renderer;
			var gui;
			var cameraControls;
			var effectController;
			
			var ambientLight, light;

			
			var mesh_points;
			
		
			var main_path = escape('{{ ply_path }}');
			init();
			animate();

			function init() {
				
				container = document.createElement( 'div' );
				document.body.appendChild( container );

				camera = new THREE.PerspectiveCamera( 35, window.innerWidth / window.innerHeight, 1, 15 );
				camera.position.set( 3, 0.15, 3 );


				

				cameraTarget = new THREE.Vector3( 0, - 0.1, 0 );

				scene = new THREE.Scene();
				scene.background = new THREE.Color( 0x72645b );
				scene.fog = new THREE.Fog( 0x72645b, 2, 15 );




				// Ground

				var plane = new THREE.Mesh(
					new THREE.PlaneBufferGeometry( 40, 40 ),
					new THREE.MeshPhongMaterial( { color: 0x999999, specular: 0x101010 } )
				);
				plane.rotation.x = - Math.PI / 2;
				plane.position.y = - 1.0;
				scene.add( plane );

				plane.receiveShadow = true;


				// PLY file

				var loader = new PLYLoader();
				
				loader.load( main_path, function ( geometry ) {

					geometry.computeVertexNormals();
					var easy_geom = new THREE.Geometry();
					easy_geom.fromBufferGeometry(geometry);
					easy_geom.normalize();

					var material = new THREE.MeshStandardMaterial( { color: 0x0055ff, flatShading: true } );
					var mesh = new THREE.Mesh( easy_geom, material );

					mesh.position.x = - 0.0;
					mesh.position.y = - 0.0;
					mesh.position.z = - 0.0;
					
					mesh.scale.multiplyScalar( 1 );


					mesh.castShadow = true;
					mesh.receiveShadow = true;
					

					let material_points = new THREE.PointsMaterial({ color:  0x0055ff, size: 0.25 });
					mesh_points = new THREE.Points(easy_geom, material_points);
					var wireframe = new THREE.WireframeGeometry( easy_geom )

					var line = new THREE.LineSegments( wireframe );
					line.material.depthTest = false;
					line.material.opacity = 0.25;
					line.material.transparent = true;
					scene.add( line );
					scene.add(mesh);
					scene.add(mesh_points);
					line.visible = false;
					mesh_points.visible = false;
					
					let mesh_names = ['base','wireframe','pointcloud']
					let meshes = [mesh,line,mesh_points]

					initGUI( mesh_names,meshes );

				} );

				

				// Lights

				scene.add( new THREE.HemisphereLight( 0x443333, 0x111122 ) );

				addShadowedLight( 1, 1, 1, 0xffffff, 1.35 );
				addShadowedLight( 0.5, 1, - 1, 0xffaa00, 1 );

				// renderer

				renderer = new THREE.WebGLRenderer( { antialias: true } );
				renderer.setPixelRatio( window.devicePixelRatio );
				renderer.setSize( window.innerWidth, window.innerHeight );
				renderer.outputEncoding = THREE.sRGBEncoding;

				renderer.shadowMap.enabled = true;

				container.appendChild( renderer.domElement );

				// CONTROLS
				cameraControls = new OrbitControls( camera, renderer.domElement );
				cameraControls.addEventListener( 'change', render );

				// stats

				stats = new Stats();
				container.appendChild( stats.dom );

				// resize

				window.addEventListener( 'resize', onWindowResize, false );

			

			}

			function addShadowedLight( x, y, z, color, intensity ) {

				var directionalLight = new THREE.DirectionalLight( color, intensity );
				directionalLight.position.set( x, y, z );
				scene.add( directionalLight );

				directionalLight.castShadow = true;

				var d = 1;
				directionalLight.shadow.camera.left = - d;
				directionalLight.shadow.camera.right = d;
				directionalLight.shadow.camera.top = d;
				directionalLight.shadow.camera.bottom = - d;

				directionalLight.shadow.camera.near = 1;
				directionalLight.shadow.camera.far = 4;

				directionalLight.shadow.mapSize.width = 1024;
				directionalLight.shadow.mapSize.height = 1024;

				directionalLight.shadow.bias = - 0.001;

			}

			function onWindowResize() {

				camera.aspect = window.innerWidth / window.innerHeight;
				camera.updateProjectionMatrix();

				renderer.setSize( window.innerWidth, window.innerHeight );

			}

			function animate() {

				requestAnimationFrame( animate );

				render();
				stats.update();

			}

			function render() {

				var timer = Date.now() * 0.0005;

				//camera.position.x = Math.sin( timer ) * 2.5;
				//camera.position.z = Math.cos( timer ) * 2.5;

				camera.lookAt( cameraTarget );

				renderer.render( scene, camera );

			}


			function initGUI( mesh_names,meshes ) {

gui = new GUI( { width: 200 } );
var layersControl = gui.addFolder( 'render modes' );
layersControl.open();

for ( var i = 0; i < mesh_names.length; i ++ ) {

	var mesh = meshes[ i ];
	var mesh_name = mesh_names[ i ];
	
	layersControl.add(mesh,'visible').name(mesh_name);

}
			}










