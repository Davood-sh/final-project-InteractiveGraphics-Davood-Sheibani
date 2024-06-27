// This function takes the translation and two rotation angles (in radians) as input arguments.
// The two rotations are applied around x and y axes.
// It returns the combined 4x4 transformation matrix as an array in column-major order.
// You can use the MatrixMult function defined in project5.html to multiply two 4x4 matrices in the same format.
function GetModelViewMatrix( translationX, translationY, translationZ, rotationX, rotationY )
{
	// [TO-DO] Modify the code below to form the transformation matrix.
	var trans = [
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		translationX, translationY, translationZ, 1
	];

	sinX = Math.sin(rotationX);
	cosX = Math.cos(rotationX);
	var rotationXMatrix = [
		1, 0, 0, 0,
		0, cosX, -sinX, 0,
		0, sinX, cosX, 0,
		0, 0, 0, 1
	];
	
	sinY = Math.sin(rotationY);
	cosY = Math.cos(rotationY);
	var rotationYMatrix = [
		cosY, 0, sinY, 0,
		0, 1, 0, 0,
		-sinY, 0, cosY, 0,
		0, 0, 0, 1
	];


	var mv =  MatrixMult(MatrixMult(trans, rotationXMatrix), rotationYMatrix);
	return mv;
}


// [TO-DO] Complete the implementation of the following class.

class MeshDrawer
{
	// The constructor is a good place for taking care of the necessary initializations.
	constructor()
	{
		// [TO-DO] initializations
		// initializations
        this.prog = InitShaderProgram(meshVS, meshFS);
        this.mvp = gl.getUniformLocation(this.prog, 'mvp');
        this.mv = gl.getUniformLocation(this.prog, 'mv');
        this.normalMatrix = gl.getUniformLocation(this.prog, 'normalMatrix');

        this.ambientLight = gl.getUniformLocation(this.prog, 'ambientLight');

        this.vertPos = gl.getAttribLocation(this.prog, 'pos');
        this.texCoordPos = gl.getAttribLocation(this.prog, 'texCoord');
        this.normalPos = gl.getAttribLocation(this.prog, 'normal');

        this.vertbuffer = gl.createBuffer();
        this.texCoordBuffer = gl.createBuffer();
        this.normalBuffer = gl.createBuffer();
        this.indexbuffer = gl.createBuffer();

        this.meshVertices = [];
        this.meshTexCoords = [];
        this.meshNormals = [];
        this.showTextureFlag = (document.getElementById("show-texture").value == 'on');
        this.textureUploaded = false;
        this.swapYZFlag = false

        this.texture = null;
	}
	
	// This method is called every time the user opens an OBJ file.
	// The arguments of this function is an array of 3D vertex positions,
	// an array of 2D texture coordinates, and an array of vertex normals.
	// Every item in these arrays is a floating point value, representing one
	// coordinate of the vertex position or texture coordinate.
	// Every three consecutive elements in the vertPos array forms one vertex
	// position and every three consecutive vertex positions form a triangle.
	// Similarly, every two consecutive elements in the texCoords array
	// form the texture coordinate of a vertex and every three consecutive 
	// elements in the normals array form a vertex normal.
	// Note that this method can be called multiple times.
	setMesh( vertPos, texCoords, normals )
	{
		// [TO-DO] Update the contents of the vertex buffer objects.
		

		// Update the contents of the vertex buffer 
        this.meshVertices = vertPos;
        this.meshTexCoords = texCoords;
		this.meshNormals = normals;

        // Bind vertex positions
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertbuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(this.meshVertices), gl.STATIC_DRAW);

        // Bind texture coordinates
        gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(this.meshTexCoords), gl.STATIC_DRAW);

		// Bind vertex normals
		gl.bindBuffer(gl.ARRAY_BUFFER, this.normalBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(this.meshNormals), gl.STATIC_DRAW);

		
	}
	
	// This method is called when the user changes the state of the
	// "Swap Y-Z Axes" checkbox. 
	// The argument is a boolean that indicates if the checkbox is checked.
	swapYZ( swap )
	{
		// [TO-DO] Set the uniform parameter(s) of the vertex shader
		const swapUniform = gl.getUniformLocation(this.prog, 'swapYZ');
        gl.useProgram(this.prog);
        gl.uniform1i(swapUniform, swap);
		this.swapYZFlag = swap;
	}
	
	// This method is called to draw the triangular mesh.
	// The arguments are the model-view-projection transformation matrixMVP,
	// the model-view transformation matrixMV, the same matrix returned
	// by the GetModelViewProjection function above, and the normal
	// transformation matrix, which is the inverse-transpose of matrixMV.
	draw(matrixMVP, matrixMV, matrixNormal, ambientLightColor) {
        gl.useProgram(this.prog);
    
        gl.uniformMatrix4fv(this.mvp, false, matrixMVP);
        gl.uniformMatrix4fv(this.mv, false, matrixMV);
        gl.uniformMatrix4fv(this.normalMatrix, false, matrixNormal);
        gl.uniform3fv(this.ambientLight, ambientLightColor);
    
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertbuffer);
        gl.vertexAttribPointer(this.vertPos, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.vertPos);
    
        gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
        gl.vertexAttribPointer(this.texCoordPos, 2, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.texCoordPos);
    
        gl.bindBuffer(gl.ARRAY_BUFFER, this.normalBuffer);
        gl.vertexAttribPointer(this.normalPos, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.normalPos);
    
        if (this.texture != null && this.showTextureFlag) {
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, this.texture);
        } else {
            const showTextureUniform = gl.getUniformLocation(this.prog, 'showTextureFlag');
            gl.uniform1i(showTextureUniform, this.showTextureFlag ? 1 : 0);
            const textureUploaded = gl.getUniformLocation(this.prog, 'textureUploaded');
            gl.uniform1i(textureUploaded, (this.texture != null));
        }
    
        this.swapYZ(this.swapYZFlag);
    
        gl.drawArrays(gl.TRIANGLES, 0, this.meshVertices.length / 3);
    }
    
	
	// This method is called to set the texture of the mesh.
	// The argument is an HTML IMG element containing the texture data.
	setTexture( img )
	{
		// [TO-DO] Bind the texture

		// You can set the texture image data using the following command.
		// Bind the texture
		console.log('Setting texture...');
		var texture = gl.createTexture(); // Create a new texture object
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, texture);

		// Set the texture image
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
		gl.generateMipmap(gl.TEXTURE_2D);

		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);

		var textureSampler = gl.getUniformLocation(this.prog, 'uSampler');
		if(textureSampler === null)
		{
			console.error('Uniform sampler not found.');
		}
		gl.useProgram(this.prog);
		// Set texture sampler uniform
		gl.uniform1i(textureSampler, 0);

		// Assign the texture object to the class property
		this.texture = texture;
		
		// Set the value of an uploaded texture
		this.textureUploaded = true;
		const textureUploaded = gl.getUniformLocation(this.prog, 'textureUploaded');
		gl.useProgram(this.prog);
		gl.uniform1i(textureUploaded, this.textureUploaded);

		// [TO-DO] Now that we have a texture, it might be a good idea to set
		// some uniform parameter(s) of the fragment shader, so that it uses the texture.
	}
	
	// This method is called when the user changes the state of the
	// "Show Texture" checkbox. 
	// The argument is a boolean that indicates if the checkbox is checked.
	showTexture( show )
	{
		// [TO-DO] set the uniform parameter(s) of the fragment shader to specify if it should use the texture.
		// set the uniform parameter(s) of the fragment shader to specify if it should use the texture.
		const showTextureUniform = gl.getUniformLocation(this.prog, 'showTextureFlag');
        gl.useProgram(this.prog);
        gl.uniform1i(showTextureUniform, show ? 1 : 0);
		const textureUploaded = gl.getUniformLocation(this.prog, 'textureUploaded');
		gl.useProgram(this.prog);
		gl.uniform1i(textureUploaded, this.textureUploaded);
		this.showTextureFlag = show;
	}
	
	// This method is called to set the incoming light direction
	setLightDir( x, y, z )
	{
		// [TO-DO] set the uniform parameter(s) of the fragment shader to specify the light direction.
		const lightDirUniform = gl.getUniformLocation(this.prog, 'lightDirection');
		
        gl.useProgram(this.prog);
        gl.uniform3f(lightDirUniform, -x, -y, z);
	}
	
	// This method is called to set the shininess of the material
	setShininess( shininess )
	{
		// [TO-DO] set the uniform parameter(s) of the fragment shader to specify the shininess.
		const shininessUniform = gl.getUniformLocation(this.prog, 'shininess');
		if (shininessUniform === null) {
			console.error('Failed to get uniform location for shininess.');
			return;
		}
        gl.useProgram(this.prog);
        gl.uniform1f(shininessUniform, shininess);
		console.log('Shininess set to:', shininess);
	}
}

// Vertex shader source code for mesh rendering
// Updated Vertex shader source code for mesh rendering
// Vertex shader source code for mesh rendering
var meshVS = `
    attribute vec3 pos;
    attribute vec2 texCoord;
    attribute vec3 normal; // Add normal attribute
    varying vec2 vTexCoord;
    varying vec3 vNormal; // Declare varying normal
    varying vec3 viewDir; // Declare varying view direction
    uniform mat4 mvp;
    uniform mat4 mv;
    uniform vec3 cameraPos; // Uniform for camera position
    uniform bool swapYZ;
    void main() {
        vec3 newPos = pos;
        if (swapYZ) {
            newPos.yz = newPos.zy;
        }
        vec4 worldPos = mv * vec4(newPos, 1.0);
        gl_Position = mvp * vec4(newPos, 1.0);
        vTexCoord = texCoord;
        vNormal = mat3(mv) * normal; // Pass normal to fragment shader
        viewDir = normalize(cameraPos - worldPos.xyz); // Calculate view direction
    }
`;




// Fragment shader source code for mesh rendering
// Fragment shader source code for mesh rendering
var meshFS = `
    precision mediump float;
    varying vec2 vTexCoord;
    varying vec3 vNormal;
    varying vec3 viewDir; // Add view direction varying
    uniform sampler2D uSampler;
    uniform bool showTextureFlag;
    uniform bool textureUploaded;
    uniform vec3 lightDirection; // Uniform for light direction
    uniform float shininess; // Uniform for shininess
    uniform vec3 ambientLight; // Ambient light uniform
    
    void main() {
        vec3 normal = normalize(vNormal);
        vec3 lightDir = normalize(lightDirection);
        vec3 viewDirection = normalize(viewDir); // View direction is now a varying variable
        
        // Compute the reflection vector
        vec3 reflectDir = reflect(-lightDir, normal);
        
        // Compute the ambient component
        vec3 ambient = ambientLight;

        // Compute the diffuse component
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 diffuse = diff * vec3(1.0); // Assuming white light source

        // Compute the specular component
        float spec = pow(max(dot(viewDirection, reflectDir), 0.0), shininess);
        vec3 specular = spec * vec3(1.0); // Assuming white light source

        // Combine the components to get the final color
        vec3 finalColor = ambient + diffuse + specular;

        // If textures are enabled and uploaded, modulate the final color with the texture color
        if (showTextureFlag && textureUploaded) {
            vec4 textureColor = texture2D(uSampler, vTexCoord);
            finalColor *= textureColor.rgb;
            gl_FragColor = vec4(finalColor, textureColor.a);
        } else {
            gl_FragColor = vec4(finalColor, 1.0);
        }
    }
`;






// This function is called for every step of the simulation.
// Its job is to advance the simulation for the given time step duration dt.
// It updates the given positions and velocities.
function SimTimeStep(dt, positions, velocities, springs, stiffness, damping, particleMass, gravity, restitution, windForce, sphere, fixedPoints) {
    var forces = Array(positions.length).fill().map(() => new Vec3(0, 0, 0));

    // Compute spring forces
    for (let spring of springs) {
        let p0 = spring.p0;
        let p1 = spring.p1;
        let restLength = spring.rest;

        let pos0 = positions[p0];
        let pos1 = positions[p1];

        let displacement = pos1.copy().sub(pos0);
        let distance = displacement.len();
        let direction = displacement.unit();

        let springForce = direction.mul(stiffness * (distance - restLength));

        forces[p0].inc(springForce);
        forces[p1].dec(springForce);
    }

    // Compute damping forces, gravity, wind force, and obstacle forces
    for (let i = 0; i < positions.length; i++) {
        if (fixedPoints.some(fp => fp.index === i)) continue; // Skip fixed points

        let velocity = velocities[i];

        // Damping force
        let dampingForce = velocity.copy().mul(-damping);
        forces[i].inc(dampingForce);

        // Gravity force
        let gravityForce = gravity.copy().mul(particleMass);
        forces[i].inc(gravityForce);

        // Wind force
        if (windForce) {
            let wind = windForce.copy();
            forces[i].inc(wind);
        }
    }

    // Add objects to the grid for collision detection
    massSpring.grid.clear();
    for (let i = 0; i < positions.length; i++) {
        massSpring.grid.addObject({ position: positions[i], index: i });
    }

    // Add objects to the grid for collision detection
    const grid = new UniformGrid(0.2);  // Initialize grid with appropriate cell size
    for (let i = 0; i < positions.length; i++) {
        grid.addObject({ position: positions[i], index: i });
    }

    // Update positions and velocities using explicit Euler integration
    for (let i = 0; i < positions.length; i++) {
        if (fixedPoints.some(fp => fp.index === i)) continue; // Skip fixed points

        let acceleration = forces[i].copy().div(particleMass);

        velocities[i].inc(acceleration.mul(dt));
        positions[i].inc(velocities[i].copy().mul(dt));

       

        // Handle collisions with the box walls
        let pos = positions[i];
        let vel = velocities[i];

        for (let axis of ['x', 'y', 'z']) {
            if (pos[axis] < -1) {
                pos[axis] = -1;
                if (vel[axis] < 0) vel[axis] *= -restitution;
            } else if (pos[axis] > 1) {
                pos[axis] = 1;
                if (vel[axis] > 0) vel[axis] *= -restitution;
            }
        }
        if (document.getElementById('show-sphere').checked) {

         // Handle collisions with the sphere
         let sphereCenter = sphere.position;
         let sphereRadius = sphere.radius;
         let distanceToSphere = pos.copy().sub(sphereCenter).len();
         
         if (distanceToSphere < sphereRadius) {
             let normal = pos.copy().sub(sphereCenter).unit();
             pos = sphereCenter.copy().add(normal.mul(sphereRadius));
             velocities[i] = velocities[i].sub(normal.mul(2 * velocities[i].dot(normal))).mul(restitution);
          }

        }

        
    }

    

    // Handle collisions for the sphere with objects if the sphere is shown
    if (document.getElementById('show-sphere').checked) {
        handleSphereCollision(sphere, positions, velocities, restitution);
    }

      // Handle self-collisions
      detectAndResolveSelfCollisions(positions, velocities, restitution, grid);

}

function detectAndResolveSelfCollisions(positions, velocities, restitution, grid) {
    const particleRadius = 0.05;  // Approximate radius of a particle for collision purposes

    // Check for and resolve collisions
    for (let i = 0; i < positions.length; i++) {
        const pos = positions[i];
        const vel = velocities[i];

        const nearbyObjects = grid.getNearbyObjects(pos);
        for (let obj of nearbyObjects) {
            if (obj.index === i) continue;  // Skip self

            const otherPos = positions[obj.index];
            const otherVel = velocities[obj.index];

            const displacement = pos.copy().sub(otherPos);
            const distance = displacement.len();

            if (distance < 2 * particleRadius) {
                const normal = displacement.unit();
                const penetrationDepth = 2 * particleRadius - distance;

                // Resolve penetration
                const correction = normal.mul(penetrationDepth / 2);
                pos.inc(correction);
                otherPos.dec(correction);

                // Calculate relative velocity
                const relativeVelocity = vel.copy().sub(otherVel);
                const dotProduct = relativeVelocity.dot(normal);

                if (dotProduct < 0) {
                    // Calculate the impulse magnitude for elastic collision
                    const impulseMagnitude = (1 + restitution) * dotProduct / 2;  // Equal mass particles
                    const impulse = normal.mul(impulseMagnitude);

                    // Update velocities
                    vel.dec(impulse);
                    otherVel.inc(impulse);
                }
            }
        }
    }
}


function handleSphereCollision(sphere, positions, velocities, restitution) {
    const sphereCenter = sphere.position;
    const sphereRadius = sphere.radius;
    const sphereMass = 0.5;  // Set to the actual mass of the sphere
    const particleMass = 0.5;  // Set to the actual mass of the particles
    const collisionFactor = 0.5;  // Factor to reduce the collision impact

    for (let i = 0; i < positions.length; i++) {
        const pos = positions[i];
        const vel = velocities[i];

        const displacement = pos.copy().sub(sphereCenter);
        const distance = displacement.len();

        if (distance < sphereRadius) {
            const normal = displacement.unit();
            const penetrationDepth = sphereRadius - distance;
            
            // Resolve penetration: Move the particle out of the sphere
            pos.set(sphereCenter.add(normal.mul(sphereRadius)));

            const relativeVelocity = vel.copy().sub(sphere.velocity);
            const dotProduct = relativeVelocity.dot(normal);

            if (dotProduct < 0) {
                // Calculate the impulse magnitude for elastic collision, considering restitution and scaling it down
                const impulseMagnitude = (1 + restitution) * dotProduct * collisionFactor / (1 / particleMass + 1 / sphereMass);
                const impulse = normal.mul(impulseMagnitude);

                // Update velocities based on scaled-down impulse
                vel.dec(impulse.div(particleMass));
                sphere.velocity.inc(impulse.div(sphereMass));
            }
        }
    }
}








class UniformGrid {
    constructor(cellSize) {
        this.cellSize = cellSize;
        this.cells = new Map();
    }

    _hashPosition(position) {
        const x = Math.floor(position.x / this.cellSize);
        const y = Math.floor(position.y / this.cellSize);
        const z = Math.floor(position.z / this.cellSize);
        return `${x},${y},${z}`;
    }

    addObject(object) {
        const hash = this._hashPosition(object.position);
        if (!this.cells.has(hash)) {
            this.cells.set(hash, []);
        }
        this.cells.get(hash).push(object);
    }

    clear() {
        this.cells.clear();
    }

    getNearbyObjects(position) {
        const hash = this._hashPosition(position);
        const nearbyObjects = [];
        for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
                for (let dz = -1; dz <= 1; dz++) {
                    const neighborHash = `${hash[0] + dx},${hash[1] + dy},${hash[2] + dz}`;
                    if (this.cells.has(neighborHash)) {
                        nearbyObjects.push(...this.cells.get(neighborHash));
                    }
                }
            }
        }
        return nearbyObjects;
    }
}

