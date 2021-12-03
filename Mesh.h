#ifndef _MONKE_MESH_H__
#define _MONKE_MESH_H__
#include <fstream>
#include <vector>
#include <string>
#include "Core.h"
#include "Triangle.h"



namespace monke
{

class Mesh
{
public:
    __host__ Mesh();
    __host__ ~Mesh();
    __host__ void LoadModel(std::string filename);
    __host__ Triangle* GetCudaTriangleList() const;
    __host__ int GetCudaTriangleSize() const;
private:
    Triangle* m_CudaTriangleList; // memory should be allocated on GPU
    std::vector<Triangle> m_Triangles;
    /* private functions */
    __host__ std::vector<int> GetIndices(const std::string& word) const;
    __host__ std::vector<std::string> GetWords(const std::string& line) const;
    __host__ void CopyToCuda();
};

Mesh::Mesh()
 :m_CudaTriangleList(nullptr)
{

}

Mesh::~Mesh()
{
    if(m_CudaTriangleList)
    {
        HANDLE_ERROR( cudaFree( m_CudaTriangleList ) );
        m_CudaTriangleList = nullptr;
    }
}

void Mesh::CopyToCuda()
{
    Triangle* triangles = (monke::Triangle*) malloc(m_Triangles.size() * sizeof(monke::Triangle));

    for(int i = 0; i < m_Triangles.size(); i++)
    {
        triangles[i].m_Vertices[0] = m_Triangles[i].m_Vertices[0];
        triangles[i].m_Vertices[1] = m_Triangles[i].m_Vertices[1];
        triangles[i].m_Vertices[2] = m_Triangles[i].m_Vertices[2];
        triangles[i].m_Color = m_Triangles[i].m_Color;
        triangles[i].CalculateNormal();
    }

    HANDLE_ERROR( cudaMalloc( (void**)&m_CudaTriangleList, m_Triangles.size() * sizeof(monke::Triangle) ) );
    HANDLE_ERROR( cudaMemcpy(m_CudaTriangleList, triangles, m_Triangles.size() * sizeof(monke::Triangle), cudaMemcpyHostToDevice) );
    free(triangles);
}

void Mesh::LoadModel(std::string filename)
{
    std::vector<Vector3f> vertices;

    std::ifstream stream(filename);
	std::string line;
	std::vector<std::string> wordVec;
	if (!stream.is_open())
	{
		printf("Could not open %s\n", filename.c_str());
        return;
	}
    printf("Reading from %s\n", filename.c_str());

    while (std::getline(stream, line))
	{
		if (line.size() != 0 && line[0] == '#') continue;
		if (line.find("v") != std::string::npos)
		{
			if ( !(line[0] == 'v' && line[1] == ' ') ) continue;
			// a vertex
			wordVec = GetWords(line);
			
			float x = std::stof(wordVec[1]);
			float y = std::stof(wordVec[2]);
			float z = std::stof(wordVec[3]);

			vertices.push_back(Vector3f(x, y, z));
		}
		else if (line.find("f") != std::string::npos)
		{
			if ( !(line[0] == 'f' && line[1] == ' ') ) continue;            
            // a face is separated by '/'
			wordVec = GetWords(line);
			
            int idx[4];
            for(int i = 1; i <= 3; i++)
            {
                std::string indexString = wordVec[i];
                idx[i] = GetIndices(indexString)[0];
            }
            m_Triangles.push_back( {vertices[ idx[1]-1 ], vertices[ idx[2]-1 ], vertices[ idx[3]-1 ] } );

		}
	}

    for(int i = 0; i < m_Triangles.size(); i++)
    {
        m_Triangles[i].PrintInfo();
    }

    printf("Number of vertices: %d\n", vertices.size());
    printf("Number of triangles: %d\n", m_Triangles.size());
	stream.close();
    CopyToCuda();
}

Triangle* Mesh::GetCudaTriangleList() const
{
    return m_CudaTriangleList;
}

int Mesh::GetCudaTriangleSize() const
{
    return m_Triangles.size();
}

std::vector<int> Mesh::GetIndices(const std::string& word) const
{
    // format vertex_index/texture_index/normal_index
    std::vector<int> indices;
    std::string s = word;//"scott/tiger/mushroom";
    std::string delimiter = "/";

    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) 
    {
        token = s.substr(0, pos);
        //std::cout << token << std::endl;
        indices.push_back( std::stoi(token) );
        s.erase(0, pos + delimiter.length());
    }

    //std::cout << s << std::endl;
    return indices;
}

std::vector<std::string> Mesh::GetWords(const std::string& line) const
{
    std::string word = "";
	std::vector<std::string> wordVec;

	for (char ch : line)
	{
		if (ch == ' ' && word != "")
		{
			wordVec.push_back(word);
			word = "";
			continue;
		}
		if(ch != ' ') word += ch;
	}
	if(word != "") wordVec.push_back(word);

	return wordVec;
}

} // end of monke

#endif