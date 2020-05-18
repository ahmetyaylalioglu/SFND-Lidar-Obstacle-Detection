/* \author Aaron Brown */
// Quiz on implementing kd tree


#ifndef KDTREE
#define KDTREE
#include "../../render/render.h"


// Structure to represent node of kd tree
struct Node
{
	std::vector<float> point;
	int id;
	Node* left;
	Node* right;

	Node(std::vector<float> arr, int setId)
	:	point(arr), id(setId), left(NULL), right(NULL)
	{}
};

struct KdTree
{
	Node* root;

	KdTree()
	: root(NULL)
	{}

	void insertHelper(Node **node,uint depth,std::vector<float> point, int id)
	{
		//tree is emptty
		if(*node == NULL)
		{
			*node = new Node(point,id);
		}
        else
		{
		uint dims = 3;
		uint cd = depth % dims;
		
		if(point[cd] < ((*node)->point[cd]))
		{
			insertHelper(&((*node)->left),depth+1,point,id);
		}
		else 
		 {
			 insertHelper(&((*node)->right),depth+1,point,id);
		 }
		}
	}


	void insert(std::vector<float> point, int id)
	{
		// TODO: Fill in this function to insert a new point into the tree
		// the function should create a new node and place correctly with in the root 
		insertHelper(&root,0,point,id);

	}
    /*KD tree seacrh implementation*/

     //the funciton created for finding 2D and 3D distance according to value of dim
    float getDistance(std::vector<float>& point1, std::vector<float>& point2,int dim)
	{
		float distance = 0.0;

		for (int i = 0; i < dim; i++)
		{
			distance = distance + pow(point1[i] - point2[i],2);
		}
		
		return sqrt(distance);
	}

	bool isInTheBox(std::vector<float>&target,Node *node, float distanceTolerance)
	{
		float leftXPoint = target[0] - distanceTolerance;
		float rightXPoint = target[0] + distanceTolerance;

		float lowerYPoint = target[1] - distanceTolerance;
		float upperYPoint = target[1] + distanceTolerance;

		float lowerZPoint = target[2] - distanceTolerance;
		float upperZPoint = target[2] + distanceTolerance;

		bool inX = false, inY = false, inZ = false,totalResult = false;

		if((leftXPoint <= node->point[0]) && (rightXPoint >= node->point[0]))
			{
				inX = true;
			}
		if((lowerYPoint <= node->point[1]) && (upperYPoint >= node->point[1]))
			{
				inY = true;
			}
		if((lowerZPoint <= node->point[2]) && (upperZPoint >= node->point[2]))
			{
				inZ = true;
			}	

		if(inX && inY && inZ)
			{
				totalResult = true;
			}	

		return totalResult;			
	}




	void searchHelper(std::vector<float> target,Node *node,uint depth,float distanceTol,std::vector<int> &ids)
	{
       if(node!=nullptr)
	   {
		   
		   int dims= 3;
		   if(isInTheBox(target,node,distanceTol) == true)
		   {
			   float dist = getDistance(target,node->point,dims);

			   if(dist <= distanceTol)
			   {
				   ids.push_back(node->id);
			   }
		   }
           
		   
		   if((target[depth%dims] + distanceTol) > node->point[depth%dims])
		   {
			   searchHelper(target, node->right, depth +1,distanceTol,ids);
		   }
		   if((target[depth%dims] -  distanceTol) < node->point[depth%dims])
		   {
			   searchHelper(target, node->left, depth +1,distanceTol,ids);
		   }
	   }

	   return;

	}

	// return a list of point ids in the tree that are within distance of target
	std::vector<int> search(std::vector<float> target, float distanceTol)
	{
		std::vector<int> ids;
		searchHelper(target,root,0,distanceTol,ids);
		return ids;
	}
	

};
#endif




