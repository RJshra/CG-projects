#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "shading_tutorial.h"

const std::string modelRelPath = "obj/bunny.obj";

ShadingTutorial::ShadingTutorial(const Options& options) : Application(options) {
	// init model
	_bunny.reset(new Model(getAssetFullPath(modelRelPath)));

	// init materials
	_ambientMaterial.reset(new AmbientMaterial);
	_ambientMaterial->ka = glm::vec3(0.03f, 0.03f, 0.03f);

	_lambertMaterial.reset(new LambertMaterial);
	_lambertMaterial->kd = glm::vec3(1.0f, 1.0f, 1.0f);

	_phongMaterial.reset(new PhongMaterial);
	_phongMaterial->ka = glm::vec3(0.03f, 0.03f, 0.03f);
	_phongMaterial->kd = glm::vec3(1.0f, 1.0f, 1.0f);
	_phongMaterial->ks = glm::vec3(1.0f, 1.0f, 1.0f);
	_phongMaterial->ns = 10.0f;

	// init shaders
	initAmbientShader();
	initLambertShader();
	initPhongShader();

	// init lights
	_ambientLight.reset(new AmbientLight);

	_directionalLight.reset(new DirectionalLight);
	_directionalLight->intensity = 0.5f;
	_directionalLight->transform.rotation = glm::angleAxis(glm::radians(45.0f), glm::normalize(glm::vec3(-1.0f)));

	_spotLight.reset(new SpotLight);
	_spotLight->intensity = 0.5f;
	_spotLight->transform.position = glm::vec3(0.0f, 0.0f, 5.0f);
	_spotLight->transform.rotation = glm::vec3(0.0f, 0.0f, 0.0f);

	// init camera
	_camera.reset(new PerspectiveCamera(glm::radians(50.0f), 1.0f * _windowWidth / _windowHeight, 0.1f, 10000.0f));
	_camera->transform.position.z = 10.0f;

	// init imgui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(_window, true);
	ImGui_ImplOpenGL3_Init();
}

ShadingTutorial::~ShadingTutorial() {
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void ShadingTutorial::initAmbientShader() {
	const char* vsCode =
		"#version 330 core\n"
		"layout(location = 0) in vec3 aPosition;\n"
		"layout(location = 1) in vec3 aNormal;\n"
		"layout(location = 2) in vec2 aTexCoord;\n"
		"uniform mat4 model;\n"
		"uniform mat4 view;\n"
		"uniform mat4 projection;\n"
		"void main() {\n"
		"	gl_Position = projection * view * model * vec4(aPosition, 1.0f);\n"
		"}\n";

	const char* fsCode =
		"#version 330 core\n"
		"out vec4 color;\n"

		"// material data structure declaration\n"
		"struct Material {\n"
		"	vec3 ka;\n"
		"};\n"

		"// ambient light data structure declaration\n"
		"struct AmbientLight {\n"
		"	vec3 color;\n"
		"	float intensity;\n"
		"};\n"
		
		"// uniform variables\n"
		"uniform Material material;\n"
		"uniform AmbientLight ambientLight;\n"
		
		"void main() {\n"
		"	vec3 ambient = material.ka * ambientLight.color * ambientLight.intensity;\n"
		"	color = vec4(ambient, 1.0f);\n"
		"}\n";

	_ambientShader.reset(new GLSLProgram);
	_ambientShader->attachVertexShader(vsCode);
	_ambientShader->attachFragmentShader(fsCode);
	_ambientShader->link();
}

void ShadingTutorial::initLambertShader() {
	const char* vsCode =
		"#version 330 core\n"
		"layout(location = 0) in vec3 aPosition;\n"
		"layout(location = 1) in vec3 aNormal;\n"
		"layout(location = 2) in vec2 aTexCoord;\n"

		"out vec3 fPosition;\n"
		"out vec3 fNormal;\n"
		
		"uniform mat4 model;\n"
		"uniform mat4 view;\n"
		"uniform mat4 projection;\n"

		"void main() {\n"
		"	fPosition = vec3(model * vec4(aPosition, 1.0f));\n"
		"	fNormal = mat3(transpose(inverse(model))) * aNormal;\n"
		"	gl_Position = projection * view * model * vec4(aPosition, 1.0f);\n"
		"}\n";

	const char* fsCode =
		"#version 330 core\n"
		"in vec3 fPosition;\n"
		"in vec3 fNormal;\n"
		"out vec4 color;\n"

		"// material data structure declaration\n"
		"struct Material {\n"
		"	vec3 kd;\n"
		"};\n"
		
		"// directional light data structure declaration\n"
		"struct DirectionalLight {\n"
		"	vec3 direction;\n"
		"	float intensity;\n"
		"	vec3 color;\n"
		"};\n"
		
		"// spot light data structure declaration\n"
		"struct SpotLight {\n"
		"	vec3 position;\n"
		"	vec3 direction;\n"
		"	float intensity;\n"
		"	vec3 color;\n"
		"	float angle;\n"
		"	float kc;\n"
		"	float kl;\n"
		"	float kq;\n"
		"};\n"
		
		"// uniform variables\n"
		"uniform Material material;\n"
		"uniform DirectionalLight directionalLight;\n"
		"uniform SpotLight spotLight;\n"

		"vec3 calcDirectionalLight(vec3 normal) {\n"
		"	vec3 lightDir = normalize(-directionalLight.direction);\n"
		"	vec3 diffuse = directionalLight.color * max(dot(lightDir, normal), 0.0f) * material.kd;\n"
		"	return directionalLight.intensity * diffuse ;\n"
		"}\n"

		"vec3 calcSpotLight(vec3 normal) {\n"
		"	vec3 lightDir = normalize(spotLight.position - fPosition);\n"
		"	float theta = acos(-dot(lightDir, normalize(spotLight.direction)));\n"
		"	if (theta > spotLight.angle) {\n"
		"		return vec3(0.0f, 0.0f, 0.0f);\n"
		"	}\n"
		"	vec3 diffuse = spotLight.color * max(dot(lightDir, normal), 0.0f) * material.kd;\n"
		"	float distance = length(spotLight.position - fPosition);\n"
		"	float attenuation = 1.0f / (spotLight.kc + spotLight.kl * distance + spotLight.kq * distance * distance);\n"
		"	return spotLight.intensity * attenuation * diffuse;\n"
		"}\n"
		
		"void main() {\n"
		"	vec3 normal = normalize(fNormal);\n"
		"	vec3 diffuse = calcDirectionalLight(normal) + calcSpotLight(normal);\n"
		"	color = vec4(diffuse, 1.0f);\n"
		"}\n";

	_lambertShader.reset(new GLSLProgram);
	_lambertShader->attachVertexShader(vsCode);
	_lambertShader->attachFragmentShader(fsCode);
	_lambertShader->link();
}

void ShadingTutorial::initPhongShader() {
	const char* vsCode =
		"#version 330 core\n"
		"layout(location = 0) in vec3 aPosition;\n"
		"layout(location = 1) in vec3 aNormal;\n"
		"layout(location = 2) in vec2 aTexCoord;\n"

		"out vec3 fPosition;\n"
		"out vec3 fNormal;\n"

		"uniform mat4 model;\n"
		"uniform mat4 view;\n"
		"uniform mat4 projection;\n"

		"void main() {\n"
		"	fPosition = vec3(model * vec4(aPosition, 1.0f));\n"
		"	fNormal = mat3(transpose(inverse(model))) * aNormal;\n"
		"	gl_Position = projection * view * model * vec4(aPosition, 1.0f);\n"
		"}\n";

	// TODO: change the shader code for phong shading
	// hint1: you can copy the fragment shader code from ambient shader to get ambient term
	// hint2: you can copy the fragment shader code from lambert shader to get diffuse term
	// hint3: you should calculate the specular term by yourself
	// hint4: add up the ambient term, diffuse term and specular term, you can get the answer
	// ------------------------------------------------------------
	const char* fsCode =
		"#version 330 core\n"
		"in vec3 fPosition;\n"
		"in vec3 fNormal;\n"
		"out vec4 color;\n"

		"uniform vec3 cameraPosition;\n"

		"// material data structure declaration\n"
		"struct Material {\n"
		"	vec3 ka;\n"
		"	vec3 kd;\n"
		"	vec3 ks;\n"
		"	float ns;\n"
		"};\n"

		"// directional light data structure declaration\n"
		"struct DirectionalLight {\n"
		"	vec3 direction;\n"
		"	float intensity;\n"
		"	vec3 color;\n"
		"};\n"

		"// spot light data structure declaration\n"
		"struct SpotLight {\n"
		"	vec3 position;\n"
		"	vec3 direction;\n"
		"	float intensity;\n"
		"	vec3 color;\n"
		"	float angle;\n"
		"	float kc;\n"
		"	float kl;\n"
		"	float kq;\n"
		"};\n"

		"// uniform variables\n"
		"uniform Material material;\n"
		"uniform DirectionalLight directionalLight;\n"
		"uniform SpotLight spotLight;\n"

		"vec3 calcDirectionalLight(vec3 normal,vec3 viewDir) {\n"

		"	vec3 ambient = directionalLight.color*material.ka;\n"

		"	vec3 lightDir = normalize(-directionalLight.direction);\n"
		"	vec3 diffuse = max(dot(lightDir, normal), 0.0f) * material.kd;\n"

		"	vec3 reflectDir=reflect(-lightDir,normal);\n"
		"	vec3 spec =pow(max(dot(viewDir,reflectDir),0.0f),material.ns)*material.ks;\n "
		"	return directionalLight.intensity * directionalLight.color*(ambient+diffuse+spec) ;\n"
		"}\n"

		"vec3 calcSpotLight(vec3 normal,vec3 viewDir) {\n"
		"	vec3 lightDir = normalize(spotLight.position - fPosition);\n"
		"	float theta = acos(-dot(lightDir, normalize(spotLight.direction)));\n"
		"	vec3 diffuse={0.0f,0.0f,0.0f};\n"
		"	if (theta <= spotLight.angle) {\n"
		"		diffuse = spotLight.color * max(dot(lightDir, normal), 0.0f) * material.kd;\n"
		"	}\n"
		"	float distance = length(spotLight.position - fPosition);\n"
		"	float attenuation = 1.0f / (spotLight.kc + spotLight.kl * distance + spotLight.kq * distance * distance);\n"

		"	vec3 ambient = spotLight.color * material.ka;\n"
		"	vec3 reflectDir=reflect(-lightDir,normal);\n"
		"	vec3 spec =directionalLight.color*pow(max(dot(reflectDir,viewDir),0.0f),material.ns)*material.ks;\n "
		"	return spotLight.intensity * attenuation * (ambient+diffuse+spec);\n"
		"}\n"

		"void main() {\n"
		"	vec3 normal = normalize(fNormal);\n"
		"	vec3 viewDir = normalize(cameraPosition-fPosition);\n"
		"	vec3 phongShader = calcDirectionalLight(normal,viewDir) + calcSpotLight(normal,viewDir);\n"
		"	color = vec4(phongShader, 1.0f);\n"
		"}\n";
	// ------------------------------------------------------------

	_phongShader.reset(new GLSLProgram);
	_phongShader->attachVertexShader(vsCode);
	_phongShader->attachFragmentShader(fsCode);
	_phongShader->link();
}

void ShadingTutorial::handleInput() {
	if (_input.keyboard.keyStates[GLFW_KEY_ESCAPE] != GLFW_RELEASE) {
		glfwSetWindowShouldClose(_window, true);
		return ;
	}
}

void ShadingTutorial::renderFrame() {
	showFpsInWindowTitle();

	glClearColor(_clearColor.r, _clearColor.g, _clearColor.b, _clearColor.a);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	switch (_renderMode) {
	case RenderMode::Ambient:
		_ambientShader->use();
		// 1. transfer mvp matrix to the shader
		_ambientShader->setUniformMat4("projection", _camera->getProjectionMatrix());
		_ambientShader->setUniformMat4("view", _camera->getViewMatrix());
		_ambientShader->setUniformMat4("model", _bunny->transform.getLocalMatrix());
		// 2. transfer material attributes to the shader
		_ambientShader->setUniformVec3("material.ka", _ambientMaterial->ka);
		// 3. transfer light attributes to the shader
		_ambientShader->setUniformVec3("ambientLight.color", _ambientLight->color);
		_ambientShader->setUniformFloat("ambientLight.intensity", _ambientLight->intensity);
		break;
	case RenderMode::Lambert:
		_lambertShader->use();
		// 1. transfer mvp matrix to the shader
		_lambertShader->setUniformMat4("projection", _camera->getProjectionMatrix());
		_lambertShader->setUniformMat4("view", _camera->getViewMatrix());
		_lambertShader->setUniformMat4("model", _bunny->transform.getLocalMatrix());
		// 2. transfer material attributes to the shader
		_lambertShader->setUniformVec3("material.kd", _lambertMaterial->kd);
		// 3. transfer light attributes to the shader
		_lambertShader->setUniformVec3("spotLight.position", _spotLight->transform.position);
		_lambertShader->setUniformVec3("spotLight.direction", _spotLight->transform.getFront());
		_lambertShader->setUniformFloat("spotLight.intensity", _spotLight->intensity);
		_lambertShader->setUniformVec3("spotLight.color", _spotLight->color);
		_lambertShader->setUniformFloat("spotLight.angle", _spotLight->angle);
		_lambertShader->setUniformFloat("spotLight.kc", _spotLight->kc);
		_lambertShader->setUniformFloat("spotLight.kl", _spotLight->kl);
		_lambertShader->setUniformFloat("spotLight.kq", _spotLight->kq);
		_lambertShader->setUniformVec3("directionalLight.direction", _directionalLight->transform.getFront());
		_lambertShader->setUniformFloat("directionalLight.intensity", _directionalLight->intensity);
		_lambertShader->setUniformVec3("directionalLight.color", _directionalLight->color);
		break;
	case RenderMode::Phong:
		_phongShader->use();
		// 1. transfer the mvp matrices to the shader
		_phongShader->setUniformMat4("projection", _camera->getProjectionMatrix());
		_phongShader->setUniformMat4("view", _camera->getViewMatrix());
		_phongShader->setUniformMat4("model", _bunny->transform.getLocalMatrix());
		
		// 2. TODO: transfer the camera position to the shader
		// write your code here
		// ----------------------------------------------------------------
		_phongShader->setUniformVec3("cameraPosition", _camera->transform.position);
		// ----------------------------------------------------------------

		
		// 3. TODO: transfer the material attributes to the shader
		// write your code here
		// -----------------------------------------------------------
		_phongShader->setUniformVec3("material.kd", _phongMaterial->kd);
		_phongShader->setUniformVec3("material.ks", _phongMaterial->ks);
		_phongShader->setUniformVec3("material.ka", _phongMaterial->ka);
		_phongShader->setUniformFloat("material.ns", _phongMaterial->ns);

		// -----------------------------------------------------------


		// 4. TODO: transfer the light attributes to the shader
		// write your code here
		// -----------------------------------------------------------
		_phongShader->setUniformVec3("spotLight.position", _spotLight->transform.position);
		_phongShader->setUniformVec3("spotLight.direction", _spotLight->transform.getFront());
		_phongShader->setUniformFloat("spotLight.intensity", _spotLight->intensity);
		_phongShader->setUniformVec3("spotLight.color", _spotLight->color);
		_phongShader->setUniformFloat("spotLight.angle", _spotLight->angle);
		_phongShader->setUniformFloat("spotLight.kc", _spotLight->kc);
		_phongShader->setUniformFloat("spotLight.kl", _spotLight->kl);
		_phongShader->setUniformFloat("spotLight.kq", _spotLight->kq);
		_phongShader->setUniformVec3("directionalLight.direction", _directionalLight->transform.getFront());
		_phongShader->setUniformFloat("directionalLight.intensity", _directionalLight->intensity);
		_phongShader->setUniformVec3("directionalLight.color", _directionalLight->color);

		// -----------------------------------------------------------

		break;
	}

	// draw the bunny
	_bunny->draw();

	// draw ui elements
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	const auto flags =
		ImGuiWindowFlags_AlwaysAutoResize |
		ImGuiWindowFlags_NoSavedSettings;

	if (!ImGui::Begin("Control Panel", nullptr, flags)) {
		ImGui::End();
	} else {
		ImGui::Text("Render Mode");
		ImGui::Separator();
		
		ImGui::RadioButton("ambient", (int*)&_renderMode, (int)(RenderMode::Ambient));
		ImGui::ColorEdit3("ka##1", (float*)&_ambientMaterial->ka);
		ImGui::NewLine();

		ImGui::RadioButton("lambert", (int*)&_renderMode, (int)(RenderMode::Lambert));
		ImGui::ColorEdit3("kd##2", (float*)&_lambertMaterial->kd);
		ImGui::NewLine();

		ImGui::RadioButton("phong", (int*)&_renderMode, (int)(RenderMode::Phong));
		ImGui::ColorEdit3("ka##3", (float*)&_phongMaterial->ka);
		ImGui::ColorEdit3("kd##3", (float*)&_phongMaterial->kd);
		ImGui::ColorEdit3("ks##3", (float*)&_phongMaterial->ks);
		ImGui::SliderFloat("ns##3", &_phongMaterial->ns, 1.0f, 50.0f);
		ImGui::NewLine();

		ImGui::Text("ambient light");
		ImGui::Separator();
		ImGui::SliderFloat("intensity##1", &_ambientLight->intensity, 0.0f, 1.0f);
		ImGui::ColorEdit3("color##1", (float*)&_ambientLight->color);
		ImGui::NewLine();

		ImGui::Text("directional light");
		ImGui::Separator();
		ImGui::SliderFloat("intensity##2", &_directionalLight->intensity, 0.0f, 1.0f);
		ImGui::ColorEdit3("color##2", (float*)&_directionalLight->color);
		ImGui::NewLine();

		ImGui::Text("spot light");
		ImGui::Separator();
		ImGui::SliderFloat("intensity##3", &_spotLight->intensity, 0.0f, 1.0f);
		ImGui::ColorEdit3("color##3", (float*)&_spotLight->color);
		ImGui::SliderFloat("angle##3", (float*)&_spotLight->angle, 0.0f, glm::radians(180.0f), "%f rad");
		ImGui::NewLine();

		ImGui::End();
	}

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}